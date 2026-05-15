# Dataset
Build barrier vehicle collision dataset from LS-DYNA d3plot files. Export the dataset in hdf5 format for training. 


conda environment:
```bash
conda env create -f dataset/environment.yml
conda activate dyna_builder
```


Build h5 dataset:
```bash
python dataset/d3plot_to_h5.py \
    --src /home/kong/datasets/barrier/fem/T_lok_F_shape_barrier_9_3_100km \
    --tmp /home/kong/datasets/barrier/tmp \
    --out /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km_30_5/output.h5 \
    --required-config configs/data/required_parts.config \
    --node-stride 30 \
    --frame-stride 5    
```

Split dataset into train/val/test:
```bash
python dataset/split_dataset.py \
    --input_dir /home/kong/datasets/barrier/h5/T_lok_F_shape_barrier_9_3_100km \
    --output_dir /home/kong/datasets/barrier/split_data/T_lok_F_shape_barrier_9_3_100km_10 \
    --context_length 10 \
    --prediction_horizon 1 \
    --frame_skip 1 \
    --split_mode temporal \
    --split_gap 10 \
    --stride 1 \
    --train_ratio 0.7 \
    --val_ratio 0.3 \
    --windows_per_file 500
```


**Parameter reference:**

- `--input_dir` — directory containing the raw simulation `.h5` files to ingest.
- `--output_dir` — where the processed `train/`, `valid/`, `test/` folders and `metadata.json` will be written.
- `--context_length` — number of past frames fed into the model as input history.
- `--prediction_horizon` — number of future frames the model is supervised to predict per window; total `window_size = context_length + prediction_horizon`.
- `--frame_skip` — temporal subsampling stride on the raw simulation (every k-th frame is kept). Changes the physical timestep: `effective_dt = raw_dt × frame_skip`. `1` means no subsampling.
- `--split_mode` — how train/val/test are carved out. `temporal` splits frames within each simulation (evaluates extrapolation in time); `by_simulation` assigns whole sims to one split (evaluates generalisation to new initial conditions).
- `--split_gap` — number of windows skipped at each train/val and val/test boundary in `temporal` mode, to prevent windows on either side from sharing frames (data leakage). Should be ≥ `context_length`.
- `--stride` — step between sliding-window start points. `1` = maximally overlapping windows (most samples, highly correlated); larger values reduce redundancy and speed up training.
- `--train_ratio` — fraction of windows (or sims, in `by_simulation` mode) assigned to training.
- `--val_ratio` — fraction assigned to validation. Test gets the remainder (`1 - train_ratio - val_ratio`).
- `--windows_per_file` — max number of window groups packed into a single output `.h5` shard before rolling over to the next file. Tunes file size vs. file count; doesn't affect dataset content.

**Implicit / auto-detected:**

- `--dt` — raw simulation timestep in seconds. Not passed here, so it's auto-detected from `/states/times` in the first input file and recorded in `metadata.json`. Pass explicitly only if you want to override the file value.
- `--max_frames_per_sim` — not used here. Truncates each sim to at most N raw frames before `frame_skip`. Useful for quick debugging on a tiny slice; omit for full data.
- Normalisation — **not** applied at this stage. Per-field mean/std are computed from training data only and stored in `metadata.json["normalization_stats"]` for the dataloader to apply on the fly.


jump to training: [README.md](../README.md)