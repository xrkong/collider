"""Time-conditioned (TC) Transolver training — history-free, non-autoregressive.

Sibling to train.py: reuses its generic config/checkpoint/W&B helpers, but
runs its own (much simpler) training loop against src/dataset_tc.py's
TCFrameDataset instead of the AR push-forward BVCSlicedDataset. See PLAN
(time-conditioned-transolver) for the full scheme.

Usage:
    python train_tc.py --experiment configs/experiments/wj11_tc.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: F401  — triggers auto-import of all registered models
from models.registry import build_model
from src.conditions import CondConfig
from src.dataset import load_or_compute_global_stats
from src.dataset_tc import build_tc_dataloader, load_or_compute_displacement_stats
from train import (
    load_config, _parse_dirs, check_git_clean, _write_meta_json, setup_wandb,
    _save_checkpoint, load_pretrained_weights, _log_best_artifact, compute_loss,
)

try:
    import wandb
    _WANDB_AVAILABLE = hasattr(wandb, "init")  # a repo-local wandb/ run-log dir can shadow the
                                                # real package as an empty namespace package
except ImportError:
    _WANDB_AVAILABLE = False


@torch.no_grad()
def validate_tc(model, val_loader, device, use_node_type: bool, accelerator: Accelerator) -> float:
    model.eval()
    loss_sum, n_batches = 0.0, 0
    for batch in val_loader:
        node_feats = batch[0].to(device)
        t_norm     = batch[1].to(device)
        target     = batch[2].to(device)
        node_type  = batch[3].to(device) if use_node_type else None

        with accelerator.autocast():
            pred = model(node_feats, node_type, t_norm)
            loss, _ = compute_loss(pred, target)
        loss_sum  += loss.item()
        n_batches += 1
    model.train()

    # Each process only sees its own DataLoader shard (accelerator.prepare
    # splits val_loader across processes) — reduce-sum before averaging so
    # multi-GPU runs report the true global val loss, not a per-process one.
    loss_sum_t  = accelerator.reduce(torch.tensor(loss_sum,  device=device), reduction="sum")
    n_batches_t = accelerator.reduce(torch.tensor(n_batches, device=device), reduction="sum")
    return (loss_sum_t / n_batches_t.clamp(min=1)).item()


def find_max_batch_size(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sample: tuple[torch.Tensor, ...],
    use_node_type: bool,
    device: torch.device,
    grad_clip: float,
    accelerator: Accelerator,
    max_candidate: int,
) -> int:
    """Largest batch size that fits one forward+backward+optimizer.step() on
    this process's GPU. Doubles from 1 until OOM, then binary-searches the gap.

    Each candidate batch is built by repeating one real (N, ...) example
    `sample` — memory scales with N and the model depth/width, not with which
    trajectory/frame the example came from, so one sample is representative.
    `.contiguous()` after `.expand()` forces genuine per-row memory (not a
    stride-0 view), matching a real batch's footprint.
    """
    node_feats_1, t_norm_1, target_1, *rest = (t.to(device) for t in sample)
    node_type_1 = rest[0] if use_node_type else None

    def probe(bs: int) -> bool:
        try:
            node_feats = node_feats_1.unsqueeze(0).expand(bs, -1, -1).contiguous()
            t_norm     = t_norm_1.unsqueeze(0).expand(bs).contiguous()
            target     = target_1.unsqueeze(0).expand(bs, -1, -1).contiguous()
            node_type  = (node_type_1.unsqueeze(0).expand(bs, -1).contiguous()
                          if use_node_type else None)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                pred = model(node_feats, node_type, t_norm)
                loss, _ = compute_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            return True
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            return False
        finally:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    last_good, bs = 0, 1
    while bs <= max_candidate:
        if probe(bs):
            last_good = bs
            bs *= 2
        else:
            break
    else:
        return last_good  # never OOM'd up to max_candidate

    if last_good == 0:
        raise RuntimeError(
            "Auto batch-size probe: even batch_size=1 does not fit in GPU memory "
            "for this model/mesh size — the model or hidden_dim/layers/slice_num is "
            "too large for this GPU. Set train.auto_batch_size=false and a small "
            "train.batch_size to get a clearer OOM, or reduce the model size."
        )

    lo, hi = last_good, bs
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if probe(mid):
            lo = mid
        else:
            hi = mid
    return lo


def train_tc(
    cfg: dict,
    git_commit: str = "unknown",
    resume_checkpoint: str | None = None,
    resume_artifact: str | None = None,
) -> float:
    """TC training loop: node_feats (rest geometry + condition) + t -> displacement-from-rest."""
    accelerator = Accelerator(mixed_precision="bf16")
    device      = accelerator.device
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]

    if not model_cfg.get("time_conditioned", False):
        raise ValueError("train_tc.py requires model.time_conditioned=true in the experiment config")

    use_node_type   = bool(data_cfg.get("node_type", False))
    has_node_emb    = model_cfg.get("num_node_types", 0) > 0 and model_cfg.get("type_emb_dim", 0) > 0
    has_partial_emb = model_cfg.get("num_node_types", 0) > 0 or model_cfg.get("type_emb_dim", 0) > 0
    if use_node_type and not has_node_emb:
        raise ValueError("data.node_type=true requires model.num_node_types>0 and model.type_emb_dim>0")
    if not use_node_type and has_partial_emb:
        raise ValueError("data.node_type=false but model.num_node_types/type_emb_dim is configured")

    accelerator.print(f"[TrainTC] Device: {device}  |  num_processes: {accelerator.num_processes}")

    cond_cfg = CondConfig(**(cfg.get("condition") or {}))
    n_cond = cond_cfg.n_cond()
    expected_in = 3 + n_cond  # reference_coords (3) + condition vector, no kinematic_bc in v1
    assert model_cfg["nnode_in_features"] == expected_in, (
        f"nnode_in_features={model_cfg['nnode_in_features']} != "
        f"3 (reference_coords) + n_cond={n_cond} = {expected_in}"
    )
    print(f"[TrainTC] Condition: enabled={list(cond_cfg.enabled)}, n_cond={n_cond}, "
          f"nnode_in_features={model_cfg['nnode_in_features']}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(model_cfg["name"], cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TrainTC] Model '{model_cfg['name']}' — {n_params:,} trainable params")

    if resume_checkpoint or resume_artifact:
        load_pretrained_weights(model, resume_checkpoint, resume_artifact, cfg)

    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # ── Data ──────────────────────────────────────────────────────────────
    train_dirs, train_barrier = _parse_dirs(data_cfg["train_dirs"])
    val_dirs,   val_barrier   = _parse_dirs(data_cfg["val_dirs"])

    run_output_dir = PROJECT_ROOT / "outputs" / "checkpoints" / cfg["name"]

    pos_stats_dict = load_or_compute_global_stats(
        train_dirs = train_dirs,
        cache_path = run_output_dir / "global_stats.json",
        fields     = ["positions"],
    )
    disp_stats = load_or_compute_displacement_stats(
        train_dirs = train_dirs,
        cache_path = run_output_dir / "global_stats_tc_displacement.json",
    )
    stats = {"positions": pos_stats_dict["positions"], "displacement": disp_stats}

    configured_batch_size = train_cfg.get("batch_size", 8)
    train_loader = build_tc_dataloader(
        cfg, train_dirs, shuffle=True,
        batch_size     = configured_batch_size,
        stats          = stats,
        barrier_params = train_barrier,
    )
    val_loader = build_tc_dataloader(
        cfg, val_dirs, shuffle=False,
        batch_size     = train_cfg.get("val_batch_size", configured_batch_size),
        stats          = stats,
        barrier_params = val_barrier,
    )

    # ── Auto batch-size probe ────────────────────────────────────────────
    # "Max out" this process's GPU memory instead of guessing a fixed number
    # (see PLAN addendum: TC max-batch-size auto-tune). CPU has no OOM concept
    # to probe against, so it's skipped there — configured_batch_size stands.
    batch_size = configured_batch_size
    if device.type == "cuda" and bool(train_cfg.get("auto_batch_size", True)):
        sample = tuple(t[0] for t in next(iter(train_loader)))
        model.to(device)
        probe_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        state_before_probe = {k: v.detach().clone() for k, v in model.state_dict().items()}

        found_bs = find_max_batch_size(
            model, probe_optimizer, sample, use_node_type, device, grad_clip,
            accelerator, max_candidate=min(4096, len(train_loader.dataset)),
        )
        model.load_state_dict(state_before_probe)  # undo the probe's gradient steps
        del probe_optimizer, state_before_probe
        torch.cuda.empty_cache()

        # Homogeneous or not, no rank should get a batch size that OOMs it —
        # take the min across processes (min(x) = -max(-x); accelerate.reduce
        # has no "min" reduction).
        batch_size = int(-accelerator.reduce(
            -torch.tensor(found_bs, device=device), reduction="max"
        ).item())
        accelerator.print(f"[TrainTC] Auto batch-size probe: {configured_batch_size} (configured) "
                           f"-> {batch_size} (max that fits, min across {accelerator.num_processes} process(es))")

        if batch_size != configured_batch_size:
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=batch_size, shuffle=True,
                num_workers=data_cfg.get("num_workers", 0), pin_memory=data_cfg.get("pin_memory", True),
            )
            val_loader = torch.utils.data.DataLoader(
                val_loader.dataset, batch_size=batch_size, shuffle=False,
                num_workers=data_cfg.get("num_workers", 0), pin_memory=data_cfg.get("pin_memory", True),
            )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # ── Optimizer schedule ────────────────────────────────────────────────
    n_epochs  = int(train_cfg["n_epochs"])
    lr        = float(train_cfg.get("lr", 1e-3))
    min_lr    = float(train_cfg.get("min_lr", lr * 0.01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # ── Checkpoint state ──────────────────────────────────────────────────
    save_dir = run_output_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = save_dir / "checkpoint_manifest.json"
    keep_top_k    = int(train_cfg.get("keep_top_k", 3))

    if accelerator.is_main_process:
        exp_meta      = _write_meta_json(save_dir, cfg)
        ckpt_history  = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        best_val_loss = min((r["val_loss"] for r in ckpt_history), default=float("inf"))
    else:
        exp_meta      = {}
        ckpt_history  = []
        best_val_loss = float("inf")

    wandb_run = setup_wandb(cfg, git_commit, exp_meta) if accelerator.is_main_process else None
    val_every = int(train_cfg.get("val_every_epochs", 1))

    accelerator.print(
        f"[TrainTC] Starting — {n_epochs} epochs | {len(train_loader)} steps/epoch/process | "
        f"batch={batch_size}/process × {accelerator.num_processes} process(es) | "
        f"lr={lr}->{min_lr} | val_every={val_every}"
    )

    step = 0
    model.train()

    try:
        for epoch in range(n_epochs):
            model.train()
            epoch_loss_sum = 0.0
            epoch_batches  = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch",
                        dynamic_ncols=True, leave=True, disable=not accelerator.is_main_process)
            for batch in pbar:
                node_feats = batch[0].to(device)                     # (B, N, 3+n_cond)
                t_norm     = batch[1].to(device)                     # (B,)
                target     = batch[2].to(device)                     # (B, N, 3)
                node_type  = batch[3].to(device) if use_node_type else None

                with accelerator.autocast():
                    pred = model(node_feats, node_type, t_norm)      # (B, N, 3)
                    loss, _ = compute_loss(pred, target)

                optimizer.zero_grad()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                step           += 1
                epoch_batches  += 1
                epoch_loss_sum += loss.item()
                lr_now = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")
                if wandb_run:
                    wandb_run.log({"train/loss": loss.item(), "lr": lr_now}, step=step)

            epoch_avg_loss = epoch_loss_sum / max(epoch_batches, 1)
            epoch_log = {"epoch": epoch + 1, "train/epoch_loss": epoch_avg_loss}

            if (epoch + 1) % val_every == 0:
                val_loss = validate_tc(model, val_loader, device, use_node_type, accelerator)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    meta_payload = {
                        "epoch": epoch + 1, "step": step, "val_loss": val_loss,
                        "git_commit": git_commit, "experiment": cfg["name"],
                    }
                    _save_checkpoint(unwrapped, save_dir / "checkpoint-latest", meta_payload)

                    ckpt_name = f"model-epoch-{epoch+1:04d}"
                    _save_checkpoint(unwrapped, save_dir / ckpt_name, meta_payload)
                    ckpt_history.append({"epoch": epoch + 1, "step": step, "val_loss": val_loss, "file": ckpt_name})
                    ckpt_history.sort(key=lambda r: r["val_loss"])
                    for stale in ckpt_history[keep_top_k:]:
                        for ext in (".safetensors", ".pt", ".json"):
                            p = save_dir / f"{stale['file']}{ext}"
                            if p.exists():
                                p.unlink()
                    ckpt_history = ckpt_history[:keep_top_k]
                    manifest_path.write_text(json.dumps(ckpt_history, indent=2))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_checkpoint(unwrapped, save_dir / "checkpoint-best", meta_payload)
                        tick = "✓ NEW BEST"
                        _log_best_artifact(wandb_run, save_dir, epoch + 1, step, val_loss, cfg)
                    else:
                        tick = ""

                    print(f"[Val] Epoch {epoch+1}/{n_epochs} | train_loss={epoch_avg_loss:.5f} | "
                          f"val_loss={val_loss:.5f} | best={best_val_loss:.5f} {tick}")

                epoch_log["val/loss"] = val_loss

            scheduler.step()
            if wandb_run:
                wandb_run.log(epoch_log, step=step)

    except KeyboardInterrupt:
        print("[TrainTC] Interrupted by user")

    accelerator.wait_for_everyone()
    accelerator.print(f"[TrainTC] Done — best val_loss: {best_val_loss:.5f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Time-Conditioned Transolver Training")
    parser.add_argument("--experiment", required=True,
                        help="Path to configs/experiments/*.yaml (must set model.time_conditioned: true)")
    parser.add_argument("--skip-git-check", action="store_true",
                        help="Skip git dirty check (debugging only)")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-checkpoint", default=None,
                        help="Local .safetensors/.pt file to initialize weights from")
    resume.add_argument("--resume-artifact", default=None,
                        help="W&B model artifact to initialize weights from, e.g. 'checkpoint-wj11_tc:best'")
    args = parser.parse_args()

    git_commit = "unknown"
    if not args.skip_git_check:
        git_commit = check_git_clean()
        print(f"[Git] Clean — commit {git_commit}")

    cfg = load_config(args.experiment)
    print(f"[Config] Loaded experiment: {cfg['name']}")

    try:
        train_tc(cfg, git_commit,
                 resume_checkpoint=args.resume_checkpoint,
                 resume_artifact=args.resume_artifact)
    finally:
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()
            print("[W&B] Run finished")


if __name__ == "__main__":
    main()
