import torch
import torch.nn as nn
from .spatial import SpatialEncoder
from .semantic import SemanticEncoder, ShapeVQEncoder
from .film import FiLMLayer

class Text2ObjectNetwork(nn.Module):
    """
    Top-level network integrating the Spatial and Semantic branches.
    Corresponds to Section 2.3: System Pipeline.
    """
    def __init__(self, text_embed_dim=512, latent_dim=128, hidden_dim=256, num_layers=4, num_embeddings=512, hashgrid=None):
        super().__init__()
        self.latent_dim = latent_dim

        # --- 1. Branch initialization ---
        # Spatial control branch (HashGrid encoding).
        hg = hashgrid or {}
        self.spatial_encoder = SpatialEncoder(
            n_levels=hg.get('n_levels', 16),
            n_features_per_level=hg.get('n_features_per_level', 2),
            log2_hashmap_size=hg.get('log2_hashmap_size', 19),
            base_resolution=hg.get('base_resolution', 16),
        )
        spatial_dim = self.spatial_encoder.output_dim

        # Semantic and latent control branches (CLIP + VQ-VAE).
        self.semantic_encoder = SemanticEncoder()
        self.vq_encoder = ShapeVQEncoder(text_embed_dim, latent_dim, num_embeddings=num_embeddings)

        # --- 2. FiLM decoder initialization ---
        self.condition_dim = text_embed_dim + latent_dim
        self.decoder_layers = nn.ModuleList()

        # First layer receives HashGrid output.
        self.decoder_layers.append(FiLMLayer(spatial_dim, hidden_dim, self.condition_dim))

        # Subsequent hidden layers.
        for _ in range(num_layers - 1):
            self.decoder_layers.append(FiLMLayer(hidden_dim, hidden_dim, self.condition_dim))

        # Final scalar SDF output layer.
        # xavier_uniform with gain=0.1 gives weight std ≈ 0.009, producing initial SDF
        # predictions with std ≈ 0.04 — large enough that ∂L/∂h is non-negligible and
        # gradients actually reach the HashGrid and FiLM layers.
        # (std=1e-4 previously killed all upstream gradients, causing constant-output collapse.)
        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, prompts, s_gt=None, z=None):
        """
        x:       3D query coordinates (Batch, N, 3)
        prompts: text descriptions List[str]
        s_gt:    ground-truth SDF (Batch, N); provide during training, None at inference.
        z:       pre-sampled latent code (Batch, latent_dim) for chunked inference.
                 If supplied, the VQ encoder is skipped entirely.

        Returns:
          Training:  sdf_pred, codebook_loss, commitment_loss
          Inference: sdf_pred, 0.0, 0.0
        """
        device = x.device
        batch_size, n_points, _ = x.shape

        # 1. Semantic Flow: frozen CLIP text feature e.
        e = self.semantic_encoder(prompts, device)

        # 2. Latent Flow: VQ-VAE encoder or pre-supplied z.
        codebook_loss   = torch.tensor(0.0, device=device)
        commitment_loss = torch.tensor(0.0, device=device)

        if s_gt is not None:
            # Training: encode SDF observations through VQ bottleneck.
            # x is detached so Eikonal gradient ∂ŝ/∂x never flows into PointNet.
            z, z_e, codebook_loss, commitment_loss, _ = self.vq_encoder(
                x.detach(), s_gt, e
            )
        elif z is None:
            # Inference (no pre-supplied z): sample a random codebook entry.
            idx = torch.randint(0, self.vq_encoder.vq.num_embeddings,
                                (batch_size,), device=device)
            z = self.vq_encoder.vq.codebook(idx)   # (B, latent_dim)

        # 3. Spatial Flow: multi-resolution HashGrid features.
        x_flat = x.view(-1, 3)
        h = self.spatial_encoder(x_flat)
        h = h.view(batch_size, n_points, -1)

        # 4. FiLM decoding: modulate spatial features with [e ‖ z].
        condition_vec = torch.cat([e, z], dim=-1)   # (B, condition_dim)
        for layer in self.decoder_layers:
            h = layer(h, condition_vec)

        # 5. Scalar SDF output.
        sdf_pred = self.output_layer(h).squeeze(-1)

        return sdf_pred, codebook_loss, commitment_loss