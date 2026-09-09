# Collider
Barrier vehicle collider simulator using machine learning methods. 

## LS-DYNA Data processing and Build dataset 
jump to [dataset](./dataset/README.md) for details.

## Training 
### Set Up Python Environment and install dependencies

[Official Conda Installation Link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
You may also need ```tmux``` to run the training in the background.

Ensure you have Python 3.11 installed (tested version). Create a new Conda environment:  
```bash
conda create --name collider 
conda activate collider

# Or run this.
pip install -r requirements.txt
```

```bash
rsync -avP --partial \
-e "ssh -i <your-ssh-key>" \
path_to_your_data \
path_to_your_remote_server
```

Build Apptainer image from definition file for HPC usage.  
```
apptainer build --fakeroot collider.sif collider.def
```


```
# Open a shell inside the container
apptainer shell --nv --bind /home/xangruik/collider:/workspace /staging/proj_iim1/xrkong/container/collider.sif
```

Dont forget line you wandb login in the container, otherwise you cannot upload your model to wandb.
```
apptainer exec /staging/proj_iim1/xrkong/container/collider.sif wandb login <YOUR_API_KEY>
```

### Connection
If you want to connect to Weitj HPC, you need to use Curtin-VPN through CISCO AnyConnect. 


### Install PyTorch (Preferably with GPU & CUDA)  
Check your GPU and CUDA compatibility before installing. The following command installs PyTorch 2.5.0 with CUDA 11.8:  
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```
For different CUDA versions, refer to [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

<!-- ### 3. Install PyTorch Geometric (PyG)  
Assuming PyTorch 2.5.0 with CUDA 11.8:  
```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
```
For other versions, refer to [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). -->

### Prepare your dataset from DYNA-style files.

Downsample d3plot files to h5 dataset for training. 
```bash
python -m dataset.build_dataset \
    --kfile  /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km/car_and_barriers.k \
    --src    /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km \
    --out    /raid/proj_iim1/xrkong/h5_fps_no_wheel/T_lok_F_shape_barrier_9_3_60km.h5 \
    --method fps \
    --seed 42 \
    --exclude-parts-config configs/data/exclude_parts_tires.yaml \
    --frame-stride 10 --n-jobs 8 \
    --gif

# Increase freame rate to 2ms (500Hz) for training, and use fps to downsample the dataset.
python -m dataset.build_dataset \
    --kfile  /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km/car_and_barriers.k \
    --src    /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km \
    --out    /raid/proj_iim1/xrkong/h5_fps_2ms_no_wheel/T_lok_F_shape_barrier_9_3_60km.h5 \
    --method fps \
    --seed 42 \
    --exclude-parts-config configs/data/exclude_parts_tires.yaml \
    --frame-stride 0 --n-jobs 8 \
    --gif
```

If you use Apptriner, run this.
```bash
cd /home/xangruik/collider
apptainer exec --bind /raid /staging/proj_iim1/xrkong/container/collider.sif \
    python -m dataset.build_dataset \
    --kfile  /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km/car_and_barriers.k \
    --src    /raid/proj_iim1/xrkong/fem/T_lok_F_shape_barrier_9_3_60km \
    --out    /raid/proj_iim1/xrkong/h5_fps_no_wheel/T_lok_F_shape_barrier_9_3_60km.h5 \
    --method fps \
    --seed 42 \
    --exclude-parts-config configs/data/exclude_parts_tires.yaml \
    --frame-stride 10 --n-jobs 8 \
    --gif
```

Analyse your downsampled dataset to check if the downsampling is correct.
```bash
apptainer exec --bind /raid /staging/proj_iim1/xrkong/container/collider.sif \
    python -m dataset.compare_downsample_fem T_lok_F_shape_barrier_9_3_100km --window-ms 0
```

If you use slurm, you can run the following command. It won't reconvert the dataset if it already exists. It is only for 500Hz dataset, so you need to change the frame stride and limit according to your needs.
```bash
sbatch --export=ALL,FRAME_STRIDE=4,FRAME_LIMIT=50 configs/hpc/build_dataset_weitj.slurm --all
```

### Training
```bash
# Train on local conda environment
python train.py --experiment configs/experiments/lc001.yaml --skip-git-check

# Train on Apptainer from a W&B artifact
apptainer exec --nv --bind /raid collider.sif accelerate launch train.py \
    --experiment configs/experiments/wj04.yaml \
    --resume-artifact "checkpoint-wj01:best"

# Train on Apptainer from a local checkpoint (no W&B needed)    
apptainer exec --nv --bind /raid collider.sif accelerate launch train.py \
    --experiment configs/experiments/wj01_2.yaml \
    --resume-checkpoint outputs/checkpoints/wj01/checkpoint-best.safetensors

# if you use slurm 
# from a local checkpoint
sbatch configs/experiments/resume_train_weitj.slurm configs/experiments/wj01_2.yaml \
    outputs/checkpoints/wj01/checkpoint-best.safetensors

# from a W&B artifact
sbatch configs/experiments/resume_train_weitj.slurm configs/experiments/wj04.yaml \
    "checkpoint-wj01:best"
```

### Barrier plate projection on xy plate 
![Barrier middle plate projection on xy plate](SPEC/lines.png) *Figure. Barrier middle plate projection on xy plate*

*Table. Barrier plate projection parameters*
| Degree | Slope m = tan(θ) | Line Equation | y-intercept (x=0) | x-intercept (y=0) |
|--------|------------------|----------------------------|-------------------|-------------------|
| −25.4° | −0.474835        | y = −0.474835x + 976.536   | 976.536           | 2056.579          |
| −20°   | −0.363970        | y = −0.363970x + 1019.672  | 1019.672          | 2801.525          |
| −15°   | −0.267949        | y = −0.267949x + 1092.698  | 1092.698          | 4078.004          |

weight part id: 2000353

### Evaluation and Rollout
After training, you can evaluate the model on the test set and perform rollouts.

```bash
python src/evaluate.py \
    --checkpoint outputs/checkpoints/exp_06/checkpoint-best.safetensors \
    --experiment configs/experiments/exp_06.yaml \
    --output-dir outputs/eval/exp_06
```

### Rollout 
```
python src/rollout.py \
        --checkpoint outputs/checkpoints/dg002/checkpoint-best.safetensors \
        --experiment configs/experiments/dg002.yaml \
        --raw-h5 /data/curtin_ciraee/curtin_xiangrui/data/h5dt_50ns_5fs_mat/T_lok_F_shape_barrier_9_3_80km/output.h5 \
        --mode both \
        --gif --gif-fps 10 \
        --gif-name dg002_80kph

sbatch configs/experiments/rollout_weitj.slurm wj04            
# just wj04
sbatch configs/experiments/rollout_weitj.slurm wj01 wj02 wj03 wj04   
# all four together, one comparison plot

```


### Uploading Model to Weights & Biases (W&B)
```
wandb artifact put \
    --name transolver_net \
    --type model \
    outputs/checkpoints/exp_collider_001/checkpoint-best.safetensors
```

# HPC user manual 

## Setup 

## environment 
install miniconda on head node, then test your script without GPU.
```bash
https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install
```

You may activate conda by running the following command:
```bash
eval "$(/data/curtin_ciraee/curtin_xiangrui/ENTER/bin/conda shell.bash hook)" && conda activate collider
```

Creat a conda env in this specific path.
```bash
conda create -p /data/curtin_ciraee/curtin_xiangrui/env/conda/collider python=3.11
```

Install packages
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


# TODO

```bash
ssh -L 9999:curtin-jupyter.hpc.dug.com:443 dug
```

# Quick Start on Weitj

1. When you get the LS-DYNA simulation cases, upload them to HPC server.
```bash
tmux new -s upload
rsync --info=progress2 -r /Users/xrkong/datasets/barrier/sep4 weitj:/raid/proj_iim1/xrkong/fem_zip
```

2. Unzip the files on HPC server.
```bash
/raid/proj_iim1/xrkong/unzip_all.sh 
```

3. Build dataset for training. 
```bash
sbatch --export=ALL,FRAME_STRIDE=4,FRAME_LIMIT=50 configs/hpc/build_dataset_weitj.slurm --all
```

4. Create a experiment yaml file for training. You can copy from existing ones and modify the parameters. `configs/experiments/wj10_r1.yaml`

5. Train the model using slurm.
```bash
sbatch --gres=gpu:2 configs/hpc/train_weitj.slurm configs/experiments/<experiment_name>.yaml
```