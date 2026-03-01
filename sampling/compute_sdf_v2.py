import gc
import csv
import json
import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_captions(csv_path: Path) -> dict:
    """
    Parse captions.tablechair.csv and return a dict mapping
    modelId -> list of description strings.
    Multiple rows with the same modelId are collected into a list.
    """
    captions = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row['modelId'].strip()
            desc = row['description'].strip()
            if desc:
                captions.setdefault(model_id, []).append(desc)
    return captions


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # Bounding Box center and bounds
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.max(bounds[1] - bounds[0])
    
    mesh.vertices -= center
    mesh.vertices /= scale  # [-0.5, 0.5]^3
    
    # [0, 1]^3
    mesh.vertices += 0.5
    return mesh

def sample_sdf(mesh: trimesh.Trimesh, num_samples: int = 50000):
    """
    Stratified SDF sampling strategy:
      40% near-surface (sigma=0.005) — tight band for sharp surface detail
      40% near-surface (sigma=0.05)  — wider band; pushes points inside thick
                                       parts of the mesh, producing positive SDF
                                       values and balancing the sign distribution
      20% uniform in [0,1]^3         — global structure / far-field
    """
    num_tight   = int(num_samples * 0.40)
    num_wide    = int(num_samples * 0.40)
    num_uniform = num_samples - num_tight - num_wide

    surface_points_t, _ = trimesh.sample.sample_surface(mesh, num_tight)
    near_tight = surface_points_t + np.random.normal(scale=0.005, size=surface_points_t.shape)

    surface_points_w, _ = trimesh.sample.sample_surface(mesh, num_wide)
    near_wide  = surface_points_w + np.random.normal(scale=0.05,  size=surface_points_w.shape)

    uniform_points = np.random.rand(num_uniform, 3)

    query_points = np.vstack([near_tight, near_wide, uniform_points])
    query_points = np.clip(query_points, 0.0, 1.0)

    # Calculate shortest distance to the surface for each point (trimesh proximity)
    try:
        sdf_values = trimesh.proximity.signed_distance(mesh, query_points)
    except Exception as e:
        print(f"SDF Calculation failed, treating as unsigned distance: {e}")
        query_obj = trimesh.proximity.ProximityQuery(mesh)
        sdf_values = query_obj.signed_distance(query_points)

    return query_points, sdf_values

def process_all_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Load captions from CSV (modelId -> [desc, ...]).
    csv_path = raw_path / "captions.tablechair.csv"
    if csv_path.exists():
        all_captions = load_captions(csv_path)
        print(f"Loaded captions for {len(all_captions)} models from {csv_path}.")
    else:
        all_captions = {}
        print(f"Warning: {csv_path} not found. Captions will not be saved.")

    # Discover all .obj files under any subfolder of raw/ (table-1, chair-2, etc.).
    obj_files = list(raw_path.glob("*/*/models/*.obj"))
    print(f"Found {len(obj_files)} 3D models. Starting preprocessing...")

    saved_captions = {}  # Captions for successfully processed models only.

    for obj_path in tqdm(obj_files):
        model_id = obj_path.parent.parent.name
        save_path = processed_path / f"{model_id}.npz"

        if save_path.exists():
            # Still record caption even if mesh was already processed.
            if model_id in all_captions and model_id not in saved_captions:
                saved_captions[model_id] = all_captions[model_id]
            continue

        mesh = None
        try:
            # Mesh
            mesh = trimesh.load(obj_path, force='mesh')

            # Remove degenerate/duplicate faces and unreferenced vertices
            # to prevent invalid-value warnings during proximity queries.
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()

            # Normalization
            mesh = normalize_mesh(mesh)
            
            # SDF Sampling
            points, sdf = sample_sdf(mesh)
            
            # save .npz
            np.savez_compressed(save_path, points=points, sdf=sdf)

            # Record caption for successfully saved model.
            if model_id in all_captions:
                saved_captions[model_id] = all_captions[model_id]
            else:
                print(f"  Note: no caption found for {model_id}.")

        except Exception as e:
            print(f"Process model {model_id} failed: {e}")
        finally:
            # Release mesh reference and free memory before next iteration.
            mesh = None
            gc.collect()

    # Write captions.json so dataset.py can load text prompts.
    captions_out = raw_path / "captions.json"
    with open(captions_out, 'w', encoding='utf-8') as f:
        json.dump(saved_captions, f, indent=2, ensure_ascii=False)
    print(f"Saved captions for {len(saved_captions)} models to {captions_out}.")

if __name__ == "__main__":
    process_all_data()