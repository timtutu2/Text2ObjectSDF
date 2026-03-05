import torch
import numpy as np
import mcubes
import trimesh
from tqdm import tqdm

def generate_mesh_from_model(
    model,
    prompt,
    device,
    resolution=128,
    chunk_size=100000,
    threshold=0.0,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    deterministic=False,
):
    model.eval()
    print(f"Generating 3D grid at resolution {resolution}^3...")
    
    x = torch.linspace(-0.5, 0.5, resolution)
    y = torch.linspace(-0.5, 0.5, resolution)
    z_axis = torch.linspace(-0.5, 0.5, resolution)
    xx, yy, zz = torch.meshgrid(x, y, z_axis, indexing='ij')
    
    grid_points = torch.stack([xx, yy, zz], dim=-1).contiguous().view(-1, 3).to(device)
    num_points = grid_points.shape[0]
    
    sdf_values = []

    # Text-only generation path:
    # text -> CLIP -> prior logits (B,T,K) -> sampled indices (B,T) -> codebook (B,T,D) -> z_agg (B,D).
    with torch.no_grad():
        token_indices, prior_logits = model.sample_tokens_from_text(
            prompts=prompt,
            device=device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            deterministic=deterministic,
        )
        z_tokens = model.lookup_tokens(token_indices)              # (B, T, D)
        z_cond = model.aggregate_shape_tokens(z_tokens)            # (B, D)
        probs = torch.softmax(prior_logits, dim=-1)                # (B, T, K)
        confidence = probs.gather(-1, token_indices.unsqueeze(-1)).mean().item()  # scalar
        print(
            f"Sampled token indices: {token_indices.tolist()} | "
            f"avg token prob: {confidence:.4f}"
        )

    print("Evaluating SDF field (chunked)...")
    with torch.no_grad():
        for i in tqdm(range(0, num_points, chunk_size)):
            chunk = grid_points[i:i+chunk_size].unsqueeze(0)
            sdf_chunk = model(chunk, mode="decode", z=z_cond)
            sdf_values.append(sdf_chunk.squeeze(0).cpu())
            
    sdf_volume = torch.cat(sdf_values, dim=0).view(resolution, resolution, resolution)
    sdf_volume = sdf_volume.contiguous().numpy()
    
    # Print stats for debugging
    print(f"SDF volume stats — min: {sdf_volume.min():.4f}  max: {sdf_volume.max():.4f}  threshold: {threshold}")

    # If the model never predicts negative SDF (e.g. underfitting), the zero-level set
    # is missing. Use the midpoint of the predicted range so we still get a mesh for
    # debugging; the surface will be at the "learned" iso-value rather than true zero.
    effective_threshold = threshold
    if sdf_volume.min() > threshold or sdf_volume.max() < threshold:
        effective_threshold = float(0.5 * (sdf_volume.min() + sdf_volume.max()))
        print(
            f"Warning: threshold {threshold} outside SDF range. "
            f"Using midpoint {effective_threshold:.4f} as iso-value (fallback; model may need more training)."
        )

    print("Running Marching Cubes algorithm...")
    vertices, triangles = mcubes.marching_cubes(sdf_volume, effective_threshold)

    if len(vertices) == 0:
        return None

    vertices = vertices / (resolution - 1) - 0.5
    mesh = trimesh.Trimesh(vertices, triangles)
    return mesh
