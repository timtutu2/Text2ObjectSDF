import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Ensure project root is on PYTHONPATH so `src.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.network import Text2ObjectNetwork
from src.utils.meshing import generate_mesh_from_model
import re


def sanitize_filename(text: str) -> str:
    """Convert a text description into a safe filename stem."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s-]+", "_", clean)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-text evaluation: generate one SDF mesh from a trained Text2ObjectSDF model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the experiment config YAML (for model + inference settings).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "model_epoch_100.pth",
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text prompt describing the object to generate. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "models",
        help="Directory to save the generated mesh (.obj).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    infer_cfg = config["inference"]

    checkpoint_path = args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    text = args.text
    if text is None:
        text = input("Enter a text description for the object: ").strip()
        if not text:
            print("Empty text prompt. Aborting.")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model --------------------------------------------------------
    print("Initialising network and loading weights...")
    model = Text2ObjectNetwork(
        text_embed_dim=model_cfg["text_embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_embeddings=model_cfg.get("num_embeddings", 512),
        hashgrid=model_cfg.get("hashgrid"),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device ,weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- Generate a single mesh -------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(text)
    output_path = args.output_dir / f"{safe_name}.obj"

    print(f"\nGenerating mesh for: '{text}'")
    mesh = generate_mesh_from_model(
        model=model,
        prompt=[text],
        device=device,
        resolution=infer_cfg["resolution"],
        chunk_size=infer_cfg["chunk_size"],
        threshold=infer_cfg["threshold"],
        cfg_scale=infer_cfg.get("cfg_scale", 3.0),
    )

    if mesh is not None:
        mesh.export(str(output_path))
        print(f"Mesh successfully saved to {output_path}")
    else:
        print("Mesh generation failed (empty surface).")


if __name__ == "__main__":
    main()
