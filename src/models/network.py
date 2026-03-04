import torch
import torch.nn as nn
from .spatial import SpatialEncoder
from .semantic import SemanticEncoder, ShapeVQEncoder, TextPrior
from .film import FiLMLayer

class Text2ObjectNetwork(nn.Module):
    """
    Text-to-shape architecture:
      1) Shape VQ encoder learns geometry tokens from (x, sdf_gt) only.
      2) Text prior predicts token distribution p(k|text).
      3) HashGrid + FiLM decoder predicts SDF conditioned only on token embedding.
    """
    def __init__(self, text_embed_dim=512, latent_dim=128, hidden_dim=256, num_layers=4, num_embeddings=512, hashgrid=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        hg = hashgrid or {}
        self.spatial_encoder = SpatialEncoder(
            n_levels=hg.get('n_levels', 16),
            n_features_per_level=hg.get('n_features_per_level', 2),
            log2_hashmap_size=hg.get('log2_hashmap_size', 19),
            base_resolution=hg.get('base_resolution', 16),
        )
        spatial_dim = self.spatial_encoder.output_dim

        self.semantic_encoder = SemanticEncoder()
        self.vq_encoder = ShapeVQEncoder(latent_dim=latent_dim, num_embeddings=num_embeddings)
        self.text_prior = TextPrior(text_embed_dim=text_embed_dim, num_embeddings=num_embeddings)

        self.condition_dim = latent_dim
        self.decoder_layers = nn.ModuleList()

        self.decoder_layers.append(FiLMLayer(spatial_dim, hidden_dim, self.condition_dim))

        for _ in range(num_layers - 1):
            self.decoder_layers.append(FiLMLayer(hidden_dim, hidden_dim, self.condition_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def decode_sdf(self, x, cond_z):
        batch_size, n_points, _ = x.shape
        x_flat = x.view(-1, 3)
        h = self.spatial_encoder(x_flat)
        h = h.view(batch_size, n_points, -1)

        for layer in self.decoder_layers:
            h = layer(h, cond_z)

        return self.output_layer(h).squeeze(-1)

    def get_prior_logits(self, prompts, device):
        e = self.semantic_encoder(prompts, device)
        return self.text_prior(e)

    @staticmethod
    def _top_k_filter(logits, top_k):
        if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
            return logits
        top_values, _ = torch.topk(logits, k=top_k, dim=-1)
        kth = top_values[..., -1, None]
        return logits.masked_fill(logits < kth, float("-inf"))

    @staticmethod
    def _top_p_filter(logits, top_p):
        if top_p is None or top_p <= 0.0 or top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return filtered_logits

    def sample_indices_from_logits(self, prior_logits, temperature=1.0, top_k=None, top_p=1.0, deterministic=False):
        if deterministic:
            return prior_logits.argmax(dim=-1)

        temperature = max(float(temperature), 1e-6)
        logits = prior_logits / temperature
        logits = self._top_k_filter(logits, top_k)
        logits = self._top_p_filter(logits, top_p)

        invalid_rows = ~torch.isfinite(logits).any(dim=-1)
        if invalid_rows.any():
            logits = logits.clone()
            logits[invalid_rows] = prior_logits[invalid_rows]

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def sample_tokens_from_text(self, prompts, device, temperature=1.0, top_k=None, top_p=1.0, deterministic=False):
        prior_logits = self.get_prior_logits(prompts, device)
        indices = self.sample_indices_from_logits(
            prior_logits=prior_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            deterministic=deterministic,
        )
        return indices, prior_logits

    def forward(
        self,
        x,
        prompts=None,
        s_gt=None,
        mode="stage1",
        indices=None,
        z=None,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        deterministic=False,
    ):
        device = x.device

        if mode == "stage1":
            if s_gt is None:
                raise ValueError("mode='stage1' requires s_gt.")
            z_q_st, codebook_loss, commitment_loss, indices_gt = self.vq_encoder(x.detach(), s_gt)
            sdf_pred = self.decode_sdf(x, z_q_st)
            return sdf_pred, codebook_loss, commitment_loss, indices_gt

        if mode == "stage2":
            if s_gt is None or prompts is None:
                raise ValueError("mode='stage2' requires both prompts and s_gt.")
            with torch.no_grad():
                indices_gt = self.vq_encoder.get_indices(x.detach(), s_gt)
            prior_logits = self.get_prior_logits(prompts, device)
            return prior_logits, indices_gt

        if mode == "decode":
            if z is None:
                if indices is None:
                    raise ValueError("mode='decode' requires either z or indices.")
                z = self.vq_encoder.vq.get_codebook_entry(indices)
            return self.decode_sdf(x, z)

        if mode == "infer_text":
            if prompts is None:
                raise ValueError("mode='infer_text' requires prompts.")
            sampled_indices, prior_logits = self.sample_tokens_from_text(
                prompts=prompts,
                device=device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                deterministic=deterministic,
            )
            cond_z = self.vq_encoder.vq.get_codebook_entry(sampled_indices)
            sdf_pred = self.decode_sdf(x, cond_z)
            return sdf_pred, sampled_indices, prior_logits

        raise ValueError(f"Unsupported mode: {mode}")
