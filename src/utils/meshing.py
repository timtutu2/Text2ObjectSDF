import torch
import numpy as np
import mcubes
import trimesh
from tqdm import tqdm

def generate_mesh_from_model(model, prompt, device, resolution=128, chunk_size=100000, threshold=0.0, cfg_scale=3.0):
    model.eval()
    print(f"Generating 3D grid at resolution {resolution}^3 with CFG scale {cfg_scale}...")
    
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    z_axis = torch.linspace(0, 1, resolution)
    xx, yy, zz = torch.meshgrid(x, y, z_axis, indexing='ij')
    
    grid_points = torch.stack([xx, yy, zz], dim=-1).contiguous().view(-1, 3).to(device)
    num_points = grid_points.shape[0]
    
    sdf_values = []

    # Derive z_cond and z_uncond from the text prior once, before the chunk loop.
    # This replaces the previous random codebook sampling and ensures the latent
    # code is actually conditioned on the user's prompt.
    with torch.no_grad():
        # Conditional: encode the user prompt and project through the text prior.
        e_cond   = model.semantic_encoder(prompt, device)
        z_prior_cond = model.text_prior(e_cond)
        _, _, _, idx_cond = model.vq_encoder.vq(z_prior_cond)
        z_cond   = model.vq_encoder.vq.codebook(idx_cond)          # (1, latent_dim)

        # Unconditional: same pipeline with an empty string — used for CFG.
        e_uncond = model.semantic_encoder([""], device)
        z_prior_uncond = model.text_prior(e_uncond)
        _, _, _, idx_uncond = model.vq_encoder.vq(z_prior_uncond)
        z_uncond = model.vq_encoder.vq.codebook(idx_uncond)        # (1, latent_dim)

    print("Evaluating SDF field (Chunking with CFG)...")
    with torch.no_grad():
        for i in tqdm(range(0, num_points, chunk_size)):
            chunk = grid_points[i:i+chunk_size].unsqueeze(0)

            # 1. Conditional prediction (with user prompt)
            sdf_cond_out, _, _, _, _ = model(chunk, prompt, s_gt=None, z=z_cond)

            # 2. Unconditional prediction (with empty string)
            sdf_uncond_out, _, _, _, _ = model(chunk, [""], s_gt=None, z=z_uncond)

            # 3. Apply Classifier-Free Guidance formula
            sdf_final = sdf_uncond_out + cfg_scale * (sdf_cond_out - sdf_uncond_out)
            
            sdf_values.append(sdf_final.squeeze(0).cpu())
            
    sdf_volume = torch.cat(sdf_values, dim=0).view(resolution, resolution, resolution)
    sdf_volume = sdf_volume.contiguous().numpy()
    
    # Print stats for debugging
    print(f"SDF volume stats — min: {sdf_volume.min():.4f}  max: {sdf_volume.max():.4f}  threshold: {threshold}")
    
    if sdf_volume.min() > threshold or sdf_volume.max() < threshold:
        print("⚠️ Warning: threshold is outside the SDF range. The mesh will be empty.")
        return None

    print("Running Marching Cubes algorithm...")
    vertices, triangles = mcubes.marching_cubes(sdf_volume, threshold)
    
    if len(vertices) == 0:
        return None
        
    vertices = vertices / (resolution - 1)
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh