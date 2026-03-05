import argparse
import inspect
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Ensure local repo root takes precedence over image-level PYTHONPATH entries (e.g. /app).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from src.data.dataset import Text2ObjectDataset
import src.models.semantic as semantic_mod
import src.models.network as network_mod

class DummySemanticEncoder(nn.Module):
    def __init__(self, text_embed_dim=512):
        super().__init__()
        self.text_embed_dim = text_embed_dim

    def forward(self, prompts, device):
        # return zeros; codebook usage test won't use it anyway
        b = len(prompts) if prompts is not None else 1
        return torch.zeros((b, self.text_embed_dim), device=device)

semantic_mod.SemanticEncoder = DummySemanticEncoder

from src.models.network import Text2ObjectNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="Check VQ codebook usage from a trained checkpoint.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/mnt/tim/text2objectsdf/checkpoints/10000_two_stage_training_stage1/stage1_model_final.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Config YAML path.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu.")
    return parser.parse_args()


def get_model_state_dict(state):
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            maybe_sd = state.get(key)
            if isinstance(maybe_sd, dict):
                state = maybe_sd
                break

    if not isinstance(state, dict):
        raise TypeError("Unsupported checkpoint format: expected a state_dict-like mapping.")

    # Support DDP checkpoints saved as "module.*"
    if any(k.startswith("module.") for k in state.keys()):
        return {k.removeprefix("module."): v for k, v in state.items()}
    return state

@torch.no_grad()
def main():
    args = parse_args()
    ckpt = args.ckpt
    cfg_path = args.config
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    model = Text2ObjectNetwork(**model_cfg).to(device)
    state = torch.load(ckpt, map_location="cpu")
    sd = get_model_state_dict(state)
    print(f"Using Text2ObjectNetwork from: {Path(inspect.getfile(network_mod)).resolve()}")
    model.load_state_dict(sd, strict=False)

    ds = Text2ObjectDataset(
        processed_dir1="/mnt/tim/data/ShapeNetCore/voxel_256_filter_div_128_solid_2",
        processed_dir2="/mnt/tim/data/ShapeNetCore/03001627_sdf",
        captions_file=str(PROJECT_ROOT / "src" / "data" / "captions_clip77.json"),
        num_points_per_batch=train_cfg["points_per_batch"],
        max_models=10000,)  
    n_eval = min(1000, len(ds))

    counter = Counter()
    for i in range(n_eval):
        x, s, _ = ds[i]                                # dataset returns (points, sdf, prompt)
        x = x.unsqueeze(0).to(device)                  # (1,N,3)
        s = s.unsqueeze(0).to(device)                  # (1,N)

        # Only run VQ encoder to get code indices.
        out = model.vq_encoder(x, s)
        indices = out[-1]  # (1,) or (1,T)
        if indices.ndim == 2:
            indices = indices.view(-1)
        counter.update(indices.detach().cpu().tolist())

    total = sum(counter.values())
    uniq = len(counter)
    top10 = counter.most_common(10)
    top1_ratio = top10[0][1] / total
    top10_ratio = sum(c for _, c in top10) / total

    print(f"Total tokens counted: {total}")
    print(f"Unique tokens used:  {uniq}")
    print(f"Top-1 token ratio:   {top1_ratio:.3f}")
    print(f"Top-10 token ratio:  {top10_ratio:.3f}")
    print("Top-10 tokens:", top10)

if __name__ == "__main__":
    main()
