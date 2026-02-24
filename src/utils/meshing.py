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
    
    random_idx = torch.randint(0, model.vq_encoder.vq.num_embeddings, (1,), device=device)
    fixed_z = model.vq_encoder.vq.codebook(random_idx)
    
    print("Evaluating SDF field (Chunking with CFG)...")
    with torch.no_grad():
        for i in tqdm(range(0, num_points, chunk_size)):
            chunk = grid_points[i:i+chunk_size].unsqueeze(0) 
            
            # 1. Conditional prediction (with user prompt)
            sdf_cond, _, _ = model(chunk, prompt, s_gt=None, z=fixed_z)
            
            # 2. Unconditional prediction (with empty string)
            sdf_uncond, _, _ = model(chunk, [""], s_gt=None, z=fixed_z)
            
            # 3. Apply Classifier-Free Guidance formula
            sdf_final = sdf_uncond + cfg_scale * (sdf_cond - sdf_uncond)
            
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