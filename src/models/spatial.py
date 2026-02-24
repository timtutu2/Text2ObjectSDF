import torch
import torch.nn as nn
import tinycudann as tcnn

class SpatialEncoder(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16):
        super().__init__()
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": 1.5,
            }
        )
        self.output_dim = self.encoder.n_output_dims

    def forward(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        x = x.contiguous().float()
        # tcnn HashGrid always outputs fp16 on CUDA; cast back to fp32 so
        # downstream nn.Linear layers (which are fp32) don't throw a dtype mismatch.
        return self.encoder(x).float()