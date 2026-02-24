import os
import re
import json
import glob
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import csv

# ---------------------------------------------------------------------------
# Config — adjust if your directory layout differs
# ---------------------------------------------------------------------------
OUTPUTS_DIR    = "outputs/models"
CAPTIONS_FILE  = "data/raw/captions.json"
RAW_DATA_DIR   = "data/raw"
CSV_OUT        = "outputs/evaluation_results.csv"
NUM_SAMPLES    = 10000


def sanitize(text: str) -> str:
    """Mirror inference.py's sanitize_filename so we can reverse-match filenames."""
    clean = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[\s-]+', '_', clean)


def build_reverse_lookup(captions_file: str) -> dict[str, str]:
    """
    Returns dict: sanitized_caption -> model_id
    Every caption variant for every model is indexed.
    """
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)

    lookup = {}
    for model_id, texts in captions.items():
        if isinstance(texts, str):
            texts = [texts]
        for t in texts:
            lookup[sanitize(t)] = model_id
    return lookup


def find_gt_mesh(model_id: str, raw_dir: str) -> str | None:
    """Search raw_dir recursively for <model_id>/models/*.obj."""
    pattern = os.path.join(raw_dir, "**", model_id, "models", "*.obj")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def normalize_mesh(mesh):
    mesh.vertices -= mesh.bounding_box.centroid
    
    max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
    
    if max_distance > 0:
        mesh.vertices /= max_distance
        
    return mesh

def compute_chamfer_distance(mesh_pred_path: str, mesh_gt_path: str,
                              num_samples: int = NUM_SAMPLES) -> float:
    try:
        mesh_pred = trimesh.load(mesh_pred_path, force='mesh')
        mesh_gt   = trimesh.load(mesh_gt_path,   force='mesh')

        if len(mesh_pred.faces) == 0 or len(mesh_gt.faces) == 0:
            return float('inf')

        mesh_pred = normalize_mesh(mesh_pred)
        mesh_gt = normalize_mesh(mesh_gt)

        pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, num_samples)
        pts_gt,   _ = trimesh.sample.sample_surface(mesh_gt,   num_samples)

        # KDTree
        dist_p2g, _ = cKDTree(pts_gt).query(pts_pred)
        dist_g2p, _ = cKDTree(pts_pred).query(pts_gt)

        # Chamfer Distance (x1000)
        return (np.mean(dist_p2g ** 2) + np.mean(dist_g2p ** 2)) * 1000

    except Exception as e:
        print(f"  Error: {e}")
        return float('inf')


def main():
    print("=" * 60)
    print("  3D Mesh Evaluation — Chamfer Distance")
    print("=" * 60)

    # Build caption → model_id reverse lookup
    if not os.path.exists(CAPTIONS_FILE):
        raise FileNotFoundError(f"Captions file not found: {CAPTIONS_FILE}")
    lookup = build_reverse_lookup(CAPTIONS_FILE)

    # Collect all generated .obj files in outputs/
    pred_files = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*.obj")))
    if not pred_files:
        print(f"No .obj files found in {OUTPUTS_DIR}/")
        return

    print(f"Found {len(pred_files)} generated mesh(es).\n")

    results = []

    for pred_path in pred_files:
        stem  = os.path.splitext(os.path.basename(pred_path))[0]
        label = stem.replace('_', ' ')

        # Match filename → model_id via sanitized captions
        model_id = lookup.get(stem)
        if model_id is None:
            print(f"[no match] {stem}")
            print(f"           Could not match to any caption in {CAPTIONS_FILE}.\n")
            results.append({"Generated": stem, "Model_ID": "N/A",
                             "GT_Found": False, "Chamfer_Distance_x1000": "N/A"})
            continue

        gt_path = find_gt_mesh(model_id, RAW_DATA_DIR)
        if gt_path is None:
            print(f"[no GT]    {stem}")
            print(f"           model_id={model_id} — no .obj found under {RAW_DATA_DIR}/\n")
            results.append({"Generated": stem, "Model_ID": model_id,
                             "GT_Found": False, "Chamfer_Distance_x1000": "N/A"})
            continue

        print(f"Evaluating: {label}")
        print(f"  pred : {pred_path}")
        print(f"  gt   : {gt_path}")
        cd = compute_chamfer_distance(pred_path, gt_path)
        status = f"{cd:.4f}" if cd != float('inf') else "failed"
        print(f"  CD   : {status}\n")

        results.append({"Generated": stem, "Model_ID": model_id,
                         "GT_Found": True, "Chamfer_Distance_x1000": cd if cd != float('inf') else "failed"})

    # Summary
    valid = [r for r in results if isinstance(r["Chamfer_Distance_x1000"], float)]
    if valid:
        scores = [r["Chamfer_Distance_x1000"] for r in valid]
        print(f"Mean CD (×1000) over {len(scores)} mesh(es): {np.mean(scores):.4f}")

    # Save CSV
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Generated", "Model_ID", "GT_Found",
                                                "Chamfer_Distance_x1000"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {CSV_OUT}")


if __name__ == "__main__":
    main()