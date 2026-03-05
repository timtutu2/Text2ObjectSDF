import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class SemanticEncoder(nn.Module):
    """
    Loads a frozen CLIP text model and extracts semantic feature vectors (e).
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        # Load tokenizer and pretrained CLIP text model.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # use_safetensors=True avoids torch.load and the transformers requirement for torch>=2.6 (CVE-2025-32434)
        self.clip_model = CLIPTextModel.from_pretrained(model_name, use_safetensors=True)

        # Freeze CLIP weights — not fine-tuned.
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def forward(self, prompts, device):
        # Tokenize and encode the text prompts.
        # CLIP has max_position_embeddings=77; truncate long captions to avoid indexing errors
        text_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = self.clip_model(**text_inputs)
        return outputs.pooler_output.float() # (Batch, 512)

class VectorQuantizer(nn.Module):
    """
    Maps a continuous encoder output z_e to its nearest codebook entry z_q.
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

    def _compute_indices(self, z_e):
        # z_e: (M, D), M can be B or (B*T)
        W = self.codebook.weight  # (K, D)
        distances = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2.0 * (z_e @ W.t())
            + W.pow(2).sum(1)
        )
        return distances.argmin(dim=1)

    def get_codebook_entry(self, indices):
        return self.codebook(indices)

    def get_indices(self, z_e):
        # z_e: (B, D) or (B, T, D)
        if z_e.dim() == 2:
            return self._compute_indices(z_e)
        if z_e.dim() == 3:
            bsz, num_tokens, dim = z_e.shape  # (B, T, D)
            flat_indices = self._compute_indices(z_e.reshape(bsz * num_tokens, dim))  # (B*T,)
            return flat_indices.view(bsz, num_tokens)  # (B, T)
        raise ValueError(f"VectorQuantizer.get_indices expects (B,D) or (B,T,D), got {tuple(z_e.shape)}")

    def forward(self, z_e):
        # z_e: (B, D) or (B, T, D)
        if z_e.dim() == 2:
            flat_shape = z_e.shape
            flat_z_e = z_e  # (B, D)
            output_is_tokenized = False
        elif z_e.dim() == 3:
            bsz, num_tokens, dim = z_e.shape  # (B, T, D)
            flat_shape = (bsz, num_tokens, dim)
            flat_z_e = z_e.reshape(bsz * num_tokens, dim)  # (B*T, D)
            output_is_tokenized = True
        else:
            raise ValueError(f"VectorQuantizer.forward expects (B,D) or (B,T,D), got {tuple(z_e.shape)}")

        flat_indices = self._compute_indices(flat_z_e)  # (M,)
        flat_z_q = self.codebook(flat_indices)  # (M, D)

        # Average over tokens (M = B or B*T), not over token count weighted by shape rank.
        codebook_loss = (flat_z_q - flat_z_e.detach()).pow(2).sum(dim=-1).mean()
        commitment_loss = (flat_z_e - flat_z_q.detach()).pow(2).sum(dim=-1).mean()

        flat_z_q_st = flat_z_e + (flat_z_q - flat_z_e).detach()  # (M, D)

        if not output_is_tokenized:
            return flat_z_q_st, codebook_loss, commitment_loss, flat_indices  # (B,D), (), (), (B,)

        z_q_st = flat_z_q_st.view(flat_shape)  # (B, T, D)
        indices = flat_indices.view(flat_shape[0], flat_shape[1])  # (B, T)
        return z_q_st, codebook_loss, commitment_loss, indices

class TextPrior(nn.Module):
    """
    Maps a CLIP text embedding to a distribution over VQ codebook indices.
    """
    def __init__(self, text_embed_dim=512, num_tokens=8, num_embeddings=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_embeddings = num_embeddings
        self.mlp = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_tokens * num_embeddings),
        )

    def forward(self, e):
        # e: (B, 512)
        logits = self.mlp(e)  # (B, T*K)
        bsz = logits.size(0)
        return logits.view(bsz, self.num_tokens, self.num_embeddings)  # (B, T, K)

class ShapeVQEncoder(nn.Module):
    """
    Encodes geometry (x + sdf) into a latent and quantizes it into shape tokens.
    """
    def __init__(self, latent_dim=128, num_embeddings=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_tokens * latent_dim)
        )
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def encode(self, x, s):
        # x: (B, N, 3), s: (B, N) or (B, N, 1)
        if s.dim() == 2:
            s = s.unsqueeze(-1)
        point_inp = torch.cat([x, s], dim=-1)  # (B, N, 4)
        point_feat = self.point_mlp(point_inp)  # (B, N, 128)
        global_feat = torch.max(point_feat, dim=1)[0]  # (B, 128)
        z_e_flat = self.latent_mlp(global_feat)  # (B, T*D)
        return z_e_flat.view(x.size(0), self.num_tokens, self.latent_dim)  # (B, T, D)

    def get_indices(self, x, s):
        z_e = self.encode(x, s)
        return self.vq.get_indices(z_e)

    def forward(self, x, s):
        z_e = self.encode(x, s)
        z_q_st, codebook_loss, commitment_loss, indices_gt = self.vq(z_e)
        return z_q_st, z_e, codebook_loss, commitment_loss, indices_gt
