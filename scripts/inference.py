import os
import yaml
import torch
import re

from src.models.network import Text2ObjectNetwork
from src.utils.meshing import generate_mesh_from_model


def sanitize_filename(text):
    """Convert a text description into a safe filename stem."""
    clean = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[\s-]+', '_', clean)


def main():
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config['model']
    infer_cfg = config['inference']

    CHECKPOINT_PATH = "checkpoints/model_final.pth"
    PROMPT_FILE     = "test_prompts.txt"

    MODELS_DIR = os.path.join("outputs", "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
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
        num_embeddings=model_cfg.get('num_embeddings', 512),
        hashgrid=model_cfg.get('hashgrid'),
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Run train.py first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(
        checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    )
    model.eval()

    # ---- Generation loop ---------------------------------------------------
    print(f"\nTotal prompts: {len(prompts)}")
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] '{prompt}'")
        safe_name = sanitize_filename(prompt)
        output_path = os.path.join(MODELS_DIR, f"{safe_name}.obj")

        mesh = generate_mesh_from_model(
            model=model,
            prompt=[prompt],
            device=device,
            resolution=infer_cfg['resolution'],
            chunk_size=infer_cfg['chunk_size'],
            threshold=infer_cfg['threshold'],
            cfg_scale=infer_cfg.get('cfg_scale', 3.0),
        )

        if mesh is not None:
            mesh.export(output_path)
            print(f"✅ Mesh successfully saved to {output_path}")
        else:
            print("  Mesh generation failed (empty surface)")


if __name__ == "__main__":
    main()