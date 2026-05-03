from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(
    train_losses: list[float],
    val_losses:   list[float],
    save_path: str | None = None,
):
    """Plot train/val loss curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        save_path:    If given, save figure to this path.

    Returns:
        matplotlib Figure (pass to ``wandb.Image()``).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="val",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    save_path: str | None = None,
):
    """Plot a confusion matrix heatmap.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: Optional list of class name strings.
        save_path:   If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    labels = class_names or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def log_figures_to_wandb(figures: dict[str, "plt.Figure"]):
    """Upload a dict of figures to W&B then close them to free memory.

    Args:
        figures: ``{"loss_curve": fig1, "confusion_matrix": fig2, ...}``

    Silently skips if W&B is not initialised.
    """
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.log({name: wandb.Image(fig) for name, fig in figures.items()})
    except ImportError:
        pass
    finally:
        for fig in figures.values():
            plt.close(fig)
