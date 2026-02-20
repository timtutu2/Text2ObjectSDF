"""
SDF Ground-Truth Visualizer for Text2ObjectSDF.

Reconstructs the zero-level set from the GT SDF of each model using
marching cubes and opens an interactive 3D viewer.

Dependencies:
    pip install trimesh pysdf scikit-image numpy

Usage:
    python sampling/visualize_sdf.py              # all 3 models sequentially
    python sampling/visualize_sdf.py --idx 1      # only model 1
    python sampling/visualize_sdf.py --res 128    # higher resolution grid
"""

import os
import glob
import argparse
import numpy as np
import trimesh
from pysdf import SDF
from skimage.measure import marching_cubes


SAMPLING_DIR = os.path.dirname(os.path.abspath(__file__))


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    verts    = mesh.vertices.copy()
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent   = (bbox_max - bbox_min).max()
    mesh     = mesh.copy()
    mesh.vertices = (verts - bbox_min) / extent
    return mesh


def build_sdf_volume(mesh: trimesh.Trimesh, res: int) -> np.ndarray:
    """
    Evaluate the GT SDF on a res^3 grid in [0, 1]^3.
    Returns a (res, res, res) float32 volume with standard sign:
      negative inside, positive outside.
    """
    lin  = np.linspace(0.0, 1.0, res, dtype=np.float32)
    xs, ys, zs = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)

    sdf_fn = SDF(mesh.vertices, mesh.faces)
    # pysdf: +inside / -outside  ->  negate for standard convention
    vol = -sdf_fn(grid).reshape(res, res, res).astype(np.float32)
    return vol


def reconstruct_and_show(obj_path: str, res: int) -> None:
    name = os.path.basename(obj_path)
    print(f"\n{'='*60}")
    print(f"Model : {name}   grid={res}^3")
    print(f"{'='*60}")

    # Load and normalize
    raw = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(raw, trimesh.Scene):
        raw = trimesh.util.concatenate(list(raw.geometry.values()))
    mesh = normalize_mesh(raw)
    print(f"  Vertices={len(mesh.vertices):,}  Faces={len(mesh.faces):,}  "
          f"Watertight={mesh.is_watertight}")

    # Build SDF volume
    print(f"  Evaluating SDF on {res}^3 grid ({res**3:,} points)...")
    vol = build_sdf_volume(mesh, res)
    print(f"  Volume SDF range: [{vol.min():.4f}, {vol.max():.4f}]")

    # Marching cubes at the zero level set
    voxel_size = 1.0 / (res - 1)
    verts, faces, normals, _ = marching_cubes(vol, level=0.0, spacing=(voxel_size,) * 3)
    recon = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    print(f"  Reconstructed mesh: {len(recon.vertices):,} verts, "
          f"{len(recon.faces):,} faces")

    # Side-by-side scene: original (blue) + reconstruction (orange)
    orig_mesh  = mesh.copy()
    orig_mesh.visual.face_colors  = [100, 149, 237, 180]   # cornflower blue
    recon.visual.face_colors      = [255, 165,   0, 180]   # orange

    scene = trimesh.Scene([orig_mesh, recon])
    print("  Opening viewer  (close window to continue)...")
    scene.show(title=f"GT SDF reconstruction — {name}  (blue=original, orange=recon)")


def main():
    parser = argparse.ArgumentParser(description="Visualize GT SDF reconstructions")
    parser.add_argument("--idx", type=int, default=None,
                        help="Model index to visualize (1, 2, or 3). "
                             "Omit to visualize all.")
    parser.add_argument("--res", type=int, default=64,
                        help="Grid resolution for marching cubes (default: 64). "
                             "Use 128 for higher fidelity (slower).")
    args = parser.parse_args()

    if args.idx is not None:
        obj_files = [os.path.join(SAMPLING_DIR, f"model_normalized_{args.idx}.obj")]
        for f in obj_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Not found: {f}")
    else:
        obj_files = sorted(
            glob.glob(os.path.join(SAMPLING_DIR, "model_normalized_*.obj"))
        )
        if not obj_files:
            raise FileNotFoundError(
                f"No model_normalized_*.obj files found in {SAMPLING_DIR}"
            )

    print(f"Visualizing {len(obj_files)} model(s) at resolution {args.res}^3.")
    for obj_path in obj_files:
        reconstruct_and_show(obj_path, args.res)

    print("\nDone.")


if __name__ == "__main__":
    main()
