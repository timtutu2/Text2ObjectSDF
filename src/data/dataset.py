import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class Text2ObjectDataset(Dataset):
    def __init__(self, processed_dir1: str = "data/processed", processed_dir2: str = "data/processed", captions_file: str = "src/data/captions.json", num_points_per_batch: int = 4096):
        self.processed_dir1 = Path(processed_dir1)
        self.processed_dir2 = Path(processed_dir2)

        self.num_points_per_batch = num_points_per_batch

        # Load caption annotations: {model_id: [caption, ...]} (model_id matches .npz stem, e.g. ue639c33f-d415-458c-8ff8-2ef68135af15)
        with open(captions_file, "r", encoding="utf-8") as f:
            self.captions_dict = json.load(f)

        # Build the dataset from the intersection of processed .npz files and captioned IDs.
        all_npz = {f.stem: f for f in self.processed_dir1.glob("*.npz")}
        all_npz.update({f.stem: f for f in self.processed_dir2.glob("*.npz")})
        
        captioned_ids = set(self.captions_dict.keys())
        npz_ids = set(all_npz.keys())

        # Preserve caption-dict order; only include IDs that have a .npz file.
        valid_ids = [mid for mid in self.captions_dict.keys() if mid in npz_ids]

        missing_captions = sorted(all_npz.keys() - captioned_ids)
        missing_npz = sorted(captioned_ids - all_npz.keys())

        if missing_captions:
            print(f"[dataset] {len(missing_captions)} processed model(s) have no caption and will be skipped.")
        if missing_npz:
            print(f"[dataset] {len(missing_npz)} captioned model(s) have no .npz file and will be skipped.")
        if not valid_ids:
            raise RuntimeError(
                f"No valid (processed + captioned) pairs found.\n"
                f"  processed_dir : {self.processed_dir}\n"
                f"  captions_file : {captions_file}"
            )

        self.model_ids = valid_ids
        self.files = [all_npz[mid] for mid in valid_ids]
        print(f"[dataset] {len(self.model_ids)} model(s) ready for training.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        model_id = self.model_ids[idx]

        # 1) Load spatial coordinates and ground-truth SDF values.
        data = np.load(file_path)
        points = data['points']  # (N, 3)
        sdf = data['sdf']        # (N,)

        # 2) Randomly subsample points to control per-object batch size.
        total_points = points.shape[0]
        if total_points > self.num_points_per_batch:
            choice = np.random.choice(total_points, self.num_points_per_batch, replace=False)
            points = points[choice]
            sdf = sdf[choice]

        # Convert to PyTorch tensors.
        points_tensor = torch.from_numpy(points).float()
        sdf_tensor = torch.from_numpy(sdf).float()

        # 3) Randomly pick one caption from the list for this model.
        captions = self.captions_dict[model_id]
        prompt = np.random.choice(captions) if isinstance(captions, list) else captions

        return points_tensor, sdf_tensor, prompt


# --- Quick dataset sanity check ---
if __name__ == "__main__":
    dataset = Text2ObjectDataset()
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        pts, sdf, text = dataset[0]
        print(f"Points shape: {pts.shape}")  # Expected: (4096, 3)
        print(f"SDF shape:    {sdf.shape}")  # Expected: (4096,)
        print(f"Prompt:       {text}")