import argparse
import os
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt

# ---- 1) Read NRRD ----
# pip install pynrrd
import nrrd


def get_spacing_from_nrrd_header(hdr, ndim=3, default=1.0):
    """
    Robustly return spacing for the LAST `ndim` spatial axes.
    Works for NRRD volumes with leading non-spatial dims (e.g., channels),
    where 'space directions' for those dims may be None/NaN.
    """
    # Prefer space directions (most reliable for NRRD)
    sd = hdr.get("space directions", None)
    if sd is not None:
        norms = []
        for v in list(sd):
            # v can be None or contain nan
            if v is None:
                norms.append(np.nan)
            else:
                v = np.array(v, dtype=np.float64).reshape(-1)
                n = float(np.linalg.norm(v))
                norms.append(n)

        norms = np.array(norms, dtype=np.float64)

        # Keep only finite, positive norms (spatial dims)
        valid = norms[np.isfinite(norms) & (norms > 0)]
        if len(valid) >= ndim:
            # take the last ndim (spatial axes)
            return tuple(valid[-ndim:].tolist())

    # Fallback to 'spacings' if present
    sp = hdr.get("spacings", None)
    if sp is not None:
        sp = np.array([np.nan if (x is None) else float(x) for x in list(sp)], dtype=np.float64)
        valid = sp[np.isfinite(sp) & (sp > 0)]
        if len(valid) >= ndim:
            return tuple(valid[-ndim:].tolist())

    # Final fallback
    return tuple([default] * ndim)
def voxel_to_sdf(voxel, spacing=(1.0, 1.0, 1.0)):
    """
    voxel: 3D array, 1=occupied/inside, 0=empty/outside
    spacing: (dx,dy,dz) in world units per voxel
    Returns:
      sdf in world units, same shape as voxel
    """
    voxel = (voxel > 0).astype(np.uint8)  # ensure binary 0/1

    # SciPy EDT supports sampling= to scale distances per-axis
    outside_dist = distance_transform_edt(voxel == 0, sampling=spacing)
    inside_dist  = distance_transform_edt(voxel == 1, sampling=spacing)

    sdf = outside_dist.astype(np.float32)
    sdf[voxel == 1] = -inside_dist[voxel == 1].astype(np.float32)
    return sdf


def sample_points_from_grid(sdf, n_points=300000, near_ratio=0.7, tau=0.1):
    """
    Produce training samples:
      - near-surface points: |sdf| < tau (if available)
      - uniform points from the entire grid
    Outputs:
      points in normalized grid coordinates [-0.5,0.5]^3
      sdf values (trilinear sampling not needed since we sample voxel centers)
    """
    D, H, W = sdf.shape

    # indices grid
    # We'll treat voxel centers and map (i,j,k) -> normalized coord in [-0.5,0.5]
    # x corresponds to i, y to j, z to k
    def idx_to_xyz(idx):
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        x = (i / (D - 1)) - 0.5
        y = (j / (H - 1)) - 0.5
        z = (k / (W - 1)) - 0.5
        return np.stack([x, y, z], axis=1).astype(np.float32)

    n_near = int(n_points * near_ratio)
    n_uni  = n_points - n_near

    # near-surface candidates
    near_mask = np.abs(sdf) <= tau
    near_idx = np.argwhere(near_mask)

    if len(near_idx) == 0:
        # fallback: if no near-surface voxels, just uniform sample
        n_near = 0
        n_uni = n_points
        near_sel = np.empty((0, 3), dtype=np.int32)
    else:
        replace = len(near_idx) < n_near
        near_sel = near_idx[np.random.choice(len(near_idx), size=n_near, replace=replace)]

    # uniform sample indices
    uni_sel = np.column_stack([
        np.random.randint(0, D, size=n_uni),
        np.random.randint(0, H, size=n_uni),
        np.random.randint(0, W, size=n_uni),
    ]).astype(np.int32)

    all_idx = np.vstack([near_sel, uni_sel])
    np.random.shuffle(all_idx)

    pts = idx_to_xyz(all_idx)
    sdf_vals = sdf[all_idx[:, 0], all_idx[:, 1], all_idx[:, 2]].astype(np.float32)
    return pts, sdf_vals


def compute_and_save(
    nrrd_path,
    out_npz_path,
    tau=0.1,
    n_points=300000,
    near_ratio=0.7,
):
    voxel, hdr = nrrd.read(nrrd_path)
    voxel = voxel.astype(np.uint8)

    # Extract spacings from header (may have more than 3 dims, e.g. a leading channel axis).
    spacing_all = get_spacing_from_nrrd_header(hdr, default=1.0)

    # If the volume has more than 3 dimensions, assume leading axes are non-spatial
    # (e.g. channels/time) and collapse them into a single 3D occupancy grid.
    if voxel.ndim > 3:
        leading_axes = tuple(range(voxel.ndim - 3))
        voxel_3d = (voxel > 0).any(axis=leading_axes).astype(np.uint8)
    else:
        voxel_3d = voxel

    # Use the last 3 spacings as the physical (x,y,z) spacing.
    if isinstance(spacing_all, (list, tuple, np.ndarray)) and len(spacing_all) >= 3:
        spacing_3d = tuple(spacing_all[-3:])
    else:
        spacing_3d = (1.0, 1.0, 1.0)

    # Sanitize spacing (replace non-finite / non-positive with default).
    spacing_3d = tuple(
        (1.0 if (not np.isfinite(s) or s <= 0) else float(s)) for s in spacing_3d
    )

    print("[NRRD] raw voxel shape:", voxel.shape)
    print("[NRRD] used 3D voxel shape:", voxel_3d.shape)
    print("[NRRD] inferred spacing (all dims):", spacing_all)
    print("[NRRD] using spacing (dx,dy,dz):", spacing_3d)
    print("[NRRD] occupied ratio (3D):", float((voxel_3d > 0).mean()))

    sdf = voxel_to_sdf(voxel_3d, spacing=spacing_3d)
    sdf_clamp = np.clip(sdf, -tau, tau).astype(np.float32)

    # sample training query points at voxel centers (normalized to [-0.5,0.5]^3)
    points, sdf_vals = sample_points_from_grid(
        sdf=sdf,
        n_points=n_points,
        near_ratio=near_ratio,
        tau=tau
    )
    sdf_vals_clamp = np.clip(sdf_vals, -tau, tau).astype(np.float32)

    np.savez_compressed(
        out_npz_path,
        points=points.astype(np.float32),
        sdf=sdf_vals.astype(np.float32),
        sdf_clamp=sdf_vals_clamp,
        sdf_grid=sdf.astype(np.float32),          # optional: keep full grid too
        sdf_grid_clamp=sdf_clamp.astype(np.float32),
        spacing=np.array(spacing_3d, dtype=np.float32),
        tau=np.float32(tau),
    )

    print("[OK] Saved:", out_npz_path)
    print("     points:", points.shape)
    print("     sdf_vals range:", float(sdf_vals.min()), float(sdf_vals.max()))
    print("     sdf_clamp range:", float(sdf_vals_clamp.min()), float(sdf_vals_clamp.max()))


def find_nrrd_in_dir(dir_path):
    """
    Return the first .nrrd file found under a model directory.

    Prefer files directly inside the folder. If none exist there, fall back to a
    recursive search so slightly different layouts still work.
    """
    root = Path(dir_path)

    direct_nrrds = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".nrrd"])
    if direct_nrrds:
        return direct_nrrds[0]

    recursive_nrrds = sorted(root.rglob("*.nrrd"))
    if recursive_nrrds:
        return recursive_nrrds[0]

    return None


def main():
    parser = argparse.ArgumentParser(description="Compute SDF samples from NRRD voxel grids.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing one subfolder per model, each with a .nrrd file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write one {model_id}.npz file per input subfolder.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Truncation threshold used to store sdf_clamp.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=300000,
        help="Number of query points to sample per model.",
    )
    parser.add_argument(
        "--near-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples drawn from the near-surface band.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Discover all .nrrd files under input_dir (any depth).
    #
    # This supports both:
    #   - "flat" layouts: input_dir/<model_id>/*.nrrd
    #   - ShapeNet-style layouts: input_dir/<synset>/<model_id>/*.nrrd
    #
    # We treat the parent directory of each .nrrd file as the model_id.
    # This matches the earlier OBJ-based pipeline where each model lived
    # in its own directory named by model_id, and also lets us safely
    # resume: if an NPZ with that model_id already exists, we skip it.
    # ------------------------------------------------------------------
    print(f"Scanning for .nrrd files under {input_dir} ...")
    nrrd_files = sorted(input_dir.rglob("*.nrrd"))
    if not nrrd_files:
        raise FileNotFoundError(f"No .nrrd files found under {input_dir}")

    print(f"Found {len(nrrd_files)} NRRD file(s) under {input_dir}")
    print(f"Config: tau={args.tau}  n_points={args.n_points:,}  near_ratio={args.near_ratio}")

    skipped = 0
    processed = 0
    for nrrd_path in nrrd_files:
        model_id = nrrd_path.parent.name
        out_npz_path = output_dir / f"{model_id}.npz"

        if out_npz_path.is_file():
            print(f"[SKIP] {model_id}: output already exists at {out_npz_path}")
            skipped += 1
            continue

        print("")
        print("=" * 60)
        print(f"Processing: {model_id}")
        print(f"  Input NRRD: {nrrd_path}")
        print(f"  Output NPZ: {out_npz_path}")
        print("=" * 60)

        compute_and_save(
            nrrd_path=str(nrrd_path),
            out_npz_path=str(out_npz_path),
            tau=args.tau,
            n_points=args.n_points,
            near_ratio=args.near_ratio,
        )
        processed += 1

    print("")
    print(f"All done. Processed={processed:,}  Skipped={skipped:,}")


if __name__ == "__main__":
    main()
