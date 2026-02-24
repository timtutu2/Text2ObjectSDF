import numpy as np
import glob

PROCESSED_DIR = "data/processed"

BUCKETS = [
    (-float("inf"), -0.10, "(-inf, -0.10)"),
    (-0.10,         -0.01, "[-0.10, -0.01)"),
    (-0.01,          0.00, "[-0.01,  0.00)"),
    ( 0.00,          0.01, "[ 0.00,  0.01)"),
    ( 0.01,          0.10, "[ 0.01,  0.10)"),
    ( 0.10,  float("inf"), "[ 0.10,  +inf)"),
]

files = glob.glob(f"{PROCESSED_DIR}/*.npz")
sdf = np.concatenate([np.load(f)["sdf"] for f in files])
n = len(sdf)

print(f"Total points : {n:,}")
print(f"Min / Max    : {sdf.min():.4f} / {sdf.max():.4f}")
print(f"Mean / Std   : {sdf.mean():.4f} / {sdf.std():.4f}")
print()

print(f"{'Bucket':<16}  {'Count':>9}  {'Ratio':>7}")
print("-" * 38)
for lo, hi, label in BUCKETS:
    count = int(((sdf >= lo) & (sdf < hi)).sum())
    print(f"{label:<16}  {count:>9,}  {count / n * 100:>6.2f}%")
