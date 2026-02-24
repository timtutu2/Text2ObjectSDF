import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class SemanticEncoder(nn.Module):
    """
    Loads a frozen CLIP text model and extracts semantic feature vectors (e).
    Corresponds to Section 2.3.2: CLIP Text Encoding.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        # Load tokenizer and pretrained CLIP text model.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip_model = CLIPTextModel.from_pretrained(model_name)

        # Freeze CLIP weights — not fine-tuned.
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, prompts, device):
        # Tokenize and encode the text prompts.
        text_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.clip_model(**text_inputs)
        return outputs.pooler_output.float() # (Batch, 512)

class VectorQuantizer(nn.Module):
    """
    Vector Quantization bottleneck.

    Maps a continuous encoder output z_e to its nearest codebook entry z_q.
    Gradients are passed straight-through from z_q back to z_e so the encoder
    is trainable despite the non-differentiable argmin.

    Losses returned (caller must weight and add to total loss):
      codebook_loss   = ||sg[z_e] - e_k||^2   (moves codebook toward encoder)
      commitment_loss = ||z_e - sg[e_k]||^2   (moves encoder toward codebook)
    """
    def __init__(self, num_embeddings=512, latent_dim=128):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim     = latent_dim

        self.codebook = nn.Embedding(num_embeddings, latent_dim)
        # Uniform init in [-1/K, 1/K] keeps initial distances well-conditioned.
        nn.init.uniform_(self.codebook.weight,
                         -1.0 / num_embeddings,
                          1.0 / num_embeddings)

    def forward(self, z_e):
        """
        z_e: (B, latent_dim)  — continuous encoder output
        Returns:
          z_q_st : (B, latent_dim)  straight-through quantized vector
          codebook_loss   : scalar
          commitment_loss : scalar
          indices         : (B,) LongTensor — nearest codebook index per sample
        """
        W = self.codebook.weight  # (K, D)
        # Squared L2 distances: ||z_e - e_k||^2 = ||z_e||^2 - 2*z_e·E^T + ||e_k||^2
        distances = (
            z_e.pow(2).sum(1, keepdim=True)   # (B, 1)
            - 2.0 * (z_e @ W.t())             # (B, K)
            + W.pow(2).sum(1)                  # (K,)
        )  # (B, K)

        indices = distances.argmin(dim=1)      # (B,)
        z_q     = self.codebook(indices)       # (B, D)

        codebook_loss   = torch.nn.functional.mse_loss(z_q, z_e.detach())
        commitment_loss = torch.nn.functional.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: forward uses z_q, backward sees z_e.
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, codebook_loss, commitment_loss, indices


class ShapeVQEncoder(nn.Module):
    """
    Shape encoder for the VQ-VAE latent branch.
    Replaces the diagonal-Gaussian posterior with a deterministic PointNet
    encoder followed by vector quantization.
    """
    def __init__(self, text_embed_dim=512, latent_dim=128, num_embeddings=512):
        super().__init__()
        # Point-wise MLP: (x, s) ∈ ℝ^4 → per-point feature.
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        # Fusion MLP: pooled geometry + CLIP text feature → continuous latent z_e.
        self.latent_mlp = nn.Sequential(
            nn.Linear(128 + text_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        # VQ bottleneck.
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def forward(self, x, s, e):
        """
        x: (B, N, 3)  — query coordinates (detached from Eikonal graph)
        s: (B, N)     — ground-truth SDF values
        e: (B, text_embed_dim)  — CLIP text feature
        Returns z_q_st, z_e, codebook_loss, commitment_loss, indices
        """
        s_expanded    = s.unsqueeze(-1)                            # (B, N, 1)
        point_inp     = torch.cat([x, s_expanded], dim=-1)        # (B, N, 4)
        point_feat    = self.point_mlp(point_inp)                  # (B, N, 128)
        global_feat   = torch.max(point_feat, dim=1)[0]            # (B, 128)

        combined      = torch.cat([global_feat, e], dim=-1)        # (B, 128+D_e)
        z_e           = self.latent_mlp(combined)                  # (B, latent_dim)

        z_q_st, codebook_loss, commitment_loss, indices = self.vq(z_e)
        return z_q_st, z_e, codebook_loss, commitment_loss, indices