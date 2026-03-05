import torch
import torch.nn as nn
from collections import Counter
from pathlib import Path
import yaml
import sys

from src.data.dataset import Text2ObjectDataset
import src.models.semantic as semantic_mod

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@torch.no_grad()
def main():
    ckpt = "/mnt/tim/text2objectsdf/checkpoints/10000_two_stage_training_stage1/stage1_model_final.pth" 
    cfg_path = "configs/default.yaml"
    device = "cuda"

    cfg = yaml.safe_load(open(cfg_path, "r"))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    model = Text2ObjectNetwork(**model_cfg).to(device)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model_state_dict", state.get("model", state))
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
        batch = ds[i]
        x = batch["x"].unsqueeze(0).to(device)         # (1,N,3)
        s = batch["sdf"].unsqueeze(0).to(device)       # (1,N)

        # 只跑 VQ encoder 拿 indices
        out = model.vq_encoder(x, s)  # 如果你還有e，改成 model.vq_encoder(x, s, e_dummy)
        # out 依你實作： (z_q_st, z_e, codebook_loss, commit_loss, indices)
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