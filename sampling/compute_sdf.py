"""
Ground-truth SDF sampling for Text2ObjectSDF.

For each model_normalized_N.obj:
  1. Load mesh and re-normalize vertices into [0, 1]^3.
  2. Sample query points:
       - Near-surface points (fine band):  surface sample + Gaussian noise sigma=0.005
       - Near-surface points (wide band):  surface sample + Gaussian noise sigma=0.05
       - Uniform points:                   uniformly drawn from [0, 1]^3
  3. Compute signed distance for every query point via pysdf (C++ BVH, headless).
     Sign convention: negative inside the mesh, positive outside  (standard SDF).
  4. Truncate SDF at threshold tau (clamp to [-tau, tau]).
  5. Save sdf_data_N.npz with keys:
       'points'    (N, 3) float32  -- query coordinates in [0, 1]^3
       'sdf'       (N,)   float32  -- raw SDF values
       'sdf_clamp' (N,)   float32  -- clamped to [-tau, tau]

Dependencies:
    pip install trimesh pysdf rtree numpy

Usage:
    python sampling/compute_sdf.py
"""

import os
import glob
import numpy as np
import trimesh
from pysdf import SDF

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLING_DIR   = os.path.dirname(os.path.abspath(__file__))

N_SURFACE_FINE = 100_000   # near-surface, tight band
N_SURFACE_WIDE = 100_000   # near-surface, wider band
N_UNIFORM      = 100_000   # uniform in [0, 1]^3

SIGMA_FINE     = 0.005     # noise std for fine band  (~0.5 % of bbox)
SIGMA_WIDE     = 0.05      # noise std for wide band  (~5 % of bbox)
TAU            = 0.1       # SDF truncation threshold

RANDOM_SEED    = 42


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Translate and uniformly scale so that all vertices fit in [0, 1]^3.
    Uses uniform scale (largest axis) to preserve aspect ratio.
    """
    verts    = mesh.vertices.copy()
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent   = (bbox_max - bbox_min).max()          # uniform scale factor
    mesh     = mesh.copy()
    mesh.vertices = (verts - bbox_min) / extent
    return mesh


def sample_query_points(
    mesh: trimesh.Trimesh,
    rng:  np.random.Generator,
) -> np.ndarray:
    """
    Return an (N, 3) float32 array of query points combining:
      - Near-surface samples at two noise levels
      - Uniform samples in [0, 1]^3
    """
    # fine near-surface band
    surf_fine, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_FINE)
    surf_fine   = surf_fine + rng.normal(0.0, SIGMA_FINE, surf_fine.shape)

    # wide near-surface band
    surf_wide, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_WIDE)
    surf_wide   = surf_wide + rng.normal(0.0, SIGMA_WIDE, surf_wide.shape)

    # uniform in unit cube
    uniform = rng.uniform(0.0, 1.0, (N_UNIFORM, 3))

    points = np.concatenate([surf_fine, surf_wide, uniform], axis=0)
    return points.astype(np.float32)


def compute_sdf_pysdf(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """
    Compute SDF values using pysdf (C++ BVH, no display required).

    pysdf returns POSITIVE values INSIDE the mesh, so we negate to get the
    standard convention: negative inside, positive outside.
    """
    sdf_fn = SDF(mesh.vertices, mesh.faces)
    # pysdf sign: +inside / -outside  ->  negate for standard SDF
    raw = sdf_fn(points)          # shape (N,)
    return (-raw).astype(np.float32)


def compute_and_save(
    obj_path: str,
    out_path: str,
    rng:      np.random.Generator,
) -> None:
    name = os.path.basename(obj_path)
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # ---- 1. Load & normalize ----
    print("  Loading mesh...")
    raw = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(list(raw.geometry.values()))
    mesh = normalize_mesh(raw)
    vmin = mesh.vertices.min(axis=0)
    vmax = mesh.vertices.max(axis=0)
    print(f"  Vertices : {len(mesh.vertices):,}    Faces: {len(mesh.faces):,}")
    print(f"  BBox     : {vmin.round(4)} -> {vmax.round(4)}")
    print(f"  Watertight: {mesh.is_watertight}")

    # ---- 2. Sample query points ----
    print("  Sampling query points...")
    points = sample_query_points(mesh, rng)
    print(f"  Total: {len(points):,}  "
          f"(fine={N_SURFACE_FINE:,}, wide={N_SURFACE_WIDE:,}, "
          f"uniform={N_UNIFORM:,})")

    # ---- 3. Compute SDF ----
    print("  Computing SDF (pysdf BVH)...")
    sdf_values = compute_sdf_pysdf(mesh, points)

    # ---- 4. Truncate ----
    sdf_clamp = np.clip(sdf_values, -TAU, TAU).astype(np.float32)

    # ---- Stats ----
    n_inside  = int((sdf_values < 0).sum())
    n_outside = int((sdf_values > 0).sum())
    n_on      = int((np.abs(sdf_values) < 1e-3).sum())
    print(f"  SDF range : [{sdf_values.min():.4f}, {sdf_values.max():.4f}]  "
          f"mean={sdf_values.mean():.4f}")
    print(f"  Inside={n_inside:,}  Outside={n_outside:,}  "
          f"Near-surface(<1e-3)={n_on:,}")

    # ---- 5. Save ----
    np.savez_compressed(
        out_path,
        points    = points,
        sdf       = sdf_values,
        sdf_clamp = sdf_clamp,
    )
    size_mb = os.path.getsize(out_path + ".npz") / 1e6
    print(f"  Saved -> {out_path}.npz  ({size_mb:.1f} MB)")


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    obj_files = sorted(
        glob.glob(os.path.join(SAMPLING_DIR, "model_normalized_*.obj"))
    )
    if not obj_files:
        raise FileNotFoundError(
            f"No model_normalized_*.obj files found in {SAMPLING_DIR}"
        )

    print(f"Found {len(obj_files)} mesh(es).")
    print(f"Config: fine={N_SURFACE_FINE:,}  wide={N_SURFACE_WIDE:,}  "
          f"uniform={N_UNIFORM:,}  sigma_fine={SIGMA_FINE}  "
          f"sigma_wide={SIGMA_WIDE}  tau={TAU}")

    for obj_path in obj_files:
        stem     = os.path.splitext(os.path.basename(obj_path))[0]   # model_normalized_1
        idx      = stem.split("_")[-1]                                # 1
        out_path = os.path.join(SAMPLING_DIR, f"sdf_data_{idx}")
        compute_and_save(obj_path, out_path, rng)

    # ---- Verify first output ----
    first_idx = os.path.splitext(os.path.basename(obj_files[0]))[0].split("_")[-1]
    verify    = os.path.join(SAMPLING_DIR, f"sdf_data_{first_idx}.npz")
    print(f"\n{'='*60}")
    print(f"Verification: {os.path.basename(verify)}")
    d = np.load(verify)
    for k, v in d.items():
        print(f"  {k:12s}: shape={v.shape}  dtype={v.dtype}  "
              f"range=[{v.min():.4f}, {v.max():.4f}]")
    print("\nAll done.")


if __name__ == "__main__":
    main()
