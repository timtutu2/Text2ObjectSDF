import argparse
import os
import re

import torch
import yaml

from src.models.network import Text2ObjectNetwork
from src.utils.meshing import generate_mesh_from_model


def sanitize_filename(text):
    """Convert a text description into a safe filename stem."""
    clean = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[\s-]+', '_', clean)


def parse_args():
    parser = argparse.ArgumentParser(description="Text-to-shape inference with token prior sampling.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override.")
    parser.add_argument("--prompt_file", type=str, default="test_prompts.txt", help="Prompt file path.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "models"), help="Mesh output dir.")
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per prompt.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k override (<=0 disables).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p override.")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax instead of sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    infer_cfg = config['inference']
    version_cfg = config.get('version', {})

    checkpoint_dir = os.path.join("checkpoints", version_cfg.get("name", "default"))
    checkpoint_path = args.checkpoint or os.path.join(checkpoint_dir, "stage2_model_final.pth")
    prompt_file = args.prompt_file

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("Prompt file is empty. Add text descriptions and try again.")
        return

    # ---- Load model --------------------------------------------------------
    print("Initialising network and loading weights...")
    model = Text2ObjectNetwork(
        text_embed_dim=model_cfg['text_embed_dim'],
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_embeddings=model_cfg.get('num_embeddings', 128),
        num_tokens=model_cfg.get('num_tokens', 8),
        hashgrid=model_cfg.get('hashgrid'),
    ).to(device)

    if not os.path.exists(checkpoint_path):
        fallback_stage1 = os.path.join(checkpoint_dir, "stage1_model_final.pth")
        if os.path.exists(fallback_stage1):
            checkpoint_path = fallback_stage1
            print(
                f"Stage2 checkpoint not found. Falling back to stage1 checkpoint: {fallback_stage1} "
                "(text prior may be untrained)."
            )
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    )
    model.eval()

    temperature = infer_cfg.get('temperature', 1.0) if args.temperature is None else args.temperature
    top_k = infer_cfg.get('top_k') if args.top_k is None else args.top_k
    top_p = infer_cfg.get('top_p', 1.0) if args.top_p is None else args.top_p
    deterministic = bool(infer_cfg.get('deterministic', False) or args.deterministic)
    num_samples = max(1, int(args.num_samples))

    # ---- Generation loop ---------------------------------------------------
    print(f"\nTotal prompts: {len(prompts)} | samples per prompt: {num_samples}")
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] '{prompt}'")
        safe_name = sanitize_filename(prompt)
        for sample_id in range(num_samples):
            suffix = f"_s{sample_id:02d}" if num_samples > 1 else ""
            output_path = os.path.join(args.output_dir, f"{safe_name}{suffix}.obj")

            mesh = generate_mesh_from_model(
                model=model,
                prompt=[prompt],
                device=device,
                resolution=infer_cfg['resolution'],
                chunk_size=infer_cfg['chunk_size'],
                threshold=infer_cfg['threshold'],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic,
            )

            if mesh is not None:
                mesh.export(output_path)
                print(f"Mesh successfully saved to {output_path}")
            else:
                print("  Mesh generation failed (empty surface)")


if __name__ == "__main__":
    main()
