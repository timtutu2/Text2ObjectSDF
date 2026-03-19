# Text2ObjectSDF

Text2ObjectSDF is a text-to-3D generation project that learns a signed distance field (SDF) from paired shape data and text captions. The pipeline uses a two-stage setup:

1. `stage1`: learn discrete shape tokens and an SDF decoder from geometry.
2. `stage2`: learn a text prior that predicts those shape tokens from text.

At inference time, the model samples shape tokens from text, decodes an SDF volume, and extracts a mesh with Marching Cubes.

## Repository Layout

- `scripts/train.py`: training entrypoint for single-GPU or `torchrun` multi-GPU training.
- `scripts/inference.py`: text-to-mesh inference entrypoint.
- `preprocessing.sh`: preprocess raw voxel/NRRD data into `.npz` files used for training.
- `configs/default.yaml`: default experiment, model, training, and inference config.
- `src/data/`: dataset code and caption metadata.
- `src/models/`: model components.
- `src/utils/meshing.py`: SDF grid evaluation and mesh extraction.

## Environment Setup

This project depends on PyTorch, `transformers`, `trimesh`, `PyMCubes`, and `tiny-cuda-nn`.

### Local setup

```bash
conda create -n text2objectsdf python=3.10 -y
conda activate text2objectsdf

sudo apt-get update
sudo apt-get install -y build-essential libspatialindex-dev

pip install -r requirements.txt
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

Notes:

- `src/models/spatial.py` uses `tinycudann`, so a CUDA-capable GPU environment is expected.
- `configs/default.yaml` enables Weights & Biases logging by default. Disable it there or set `WANDB_API_KEY` in your environment.

### Docker setup

The repo also includes a `Dockerfile` that installs CUDA dependencies and `tiny-cuda-nn`.

```bash
docker build -t text2objectsdf .
```

You can also pull the prebuilt image directly:

```bash
docker pull timttu/text2objectsdf:v5
```

## Data Preparation

Training expects preprocessed `.npz` files plus caption annotations.

Download the text-shape dataset here:

http://text2shape.stanford.edu/

The preprocessing script currently converts NRRD files into `.npz` samples:

In this codebase, preprocessing turns voxelized `.nrrd` shape files into `.npz` files that store sampled 3D points and their corresponding clamped SDF values. These `.npz` files are the training format consumed by the dataset loader in `src/data/dataset.py`.

```bash
bash preprocessing.sh
```

Important:

- `preprocessing.sh` contains hardcoded input/output paths under `your path/...`.
- `scripts/train.py` also hardcodes the processed dataset path.
- If you are running on a different machine or directory layout, update those paths before training.

## Training

The default config is:

```bash
configs/default.yaml
```

Checkpoints are saved under:

```bash
checkpoints/10000_two_stage_training_multi_token_stage2_v4/
```

because `configs/default.yaml` sets:

```yaml
version:
  name: "10000_two_stage_training_multi_token_stage2_v4"
```

### Stage 1

Train the shape tokenizer + decoder:

```bash
python scripts/train.py --config configs/default.yaml --stage stage1
```

### Stage 2

Train the text prior from a stage 1 checkpoint:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --stage stage2 \
  --resume checkpoints/10000_two_stage_training_multi_token_stage2_v4/stage1_model_final.pth
```

### Multi-GPU training

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --stage stage1
```

and similarly for stage 2:

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/default.yaml \
  --stage stage2 \
  --resume checkpoints/10000_two_stage_training_multi_token_stage2_v4/stage1_model_final.pth
```

## Final Checkpoint

The final checkpoint is available here:

https://drive.google.com/drive/folders/10VOdCAxoJrd2Qob0haF-S-7UoJLt45BP?usp=sharing

You can run inference directly with that checkpoint.

Recommended options:

- Download `stage2_model_final.pth` and place it at:
  `checkpoints/10000_two_stage_training_multi_token_stage2_v4/stage2_model_final.pth`
- Or keep it anywhere and pass the path with `--checkpoint`.

## Inference

`scripts/inference.py` reads prompts from a text file, one prompt per line, and exports meshes as `.obj`.

Example prompt file:

```bash
printf "a wooden chair with four legs\nan office chair with wheels\n" > prompts.txt
```

Run inference with the downloaded checkpoint:

```bash
python scripts/inference.py \
  --config configs/default.yaml \
  --checkpoint /path/to/stage2_model_final.pth \
  --prompt_file prompts.txt \
  --output_dir outputs/models \
  --num_samples 1
```

Generated meshes will be written to:

```bash
outputs/models/
```

Useful inference flags:

- `--temperature`: sampling temperature override.
- `--top_k`: top-k token filtering.
- `--top_p`: nucleus sampling.
- `--deterministic`: use argmax instead of stochastic sampling.
- `--num_samples`: generate multiple meshes per prompt.

If `--checkpoint` is not provided, inference defaults to:

```bash
checkpoints/10000_two_stage_training_multi_token_stage2_v4/stage2_model_final.pth
```

If that file is missing, the script falls back to `stage1_model_final.pth`, but text-conditioned generation may be worse because the text prior may be untrained.

## Outputs

- Training checkpoints: `checkpoints/<version_name>/`
- Logs: `your path/text2objectsdf/logs`
- Generated meshes: `outputs/models/*.obj`

## Notes

- The current training code uses the cleaned caption file `src/data/captions_clip77_clean.json`.
- The current dataset loader samples one caption per object and randomly subsamples SDF points each iteration.
- The current implementation is written around the directory structure used in the project environment, so path cleanup is the first thing to change if you want to make the repo portable.
