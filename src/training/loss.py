import torch
import torch.nn as nn
import torch.nn.functional as F

class Text2ObjectLoss(nn.Module):
    """
    Loss for SDF reconstruction + VQ regularization (+ optional prior CE).
    """
    def __init__(self, truncation_dist=0.1, lambda_sdf=1.0, lambda_codebook=1.0,
                 commitment_cost=0.25, lambda_eik=0.1, lambda_prior=1.0,
                 lambda_far=0.1):
        super().__init__()
        self.tau              = truncation_dist
        self.lambda_sdf       = lambda_sdf        # weight for truncated SDF loss (try 2–5 if inference SDF is all-positive)
        self.lambda_codebook  = lambda_codebook   # weight for codebook loss
        self.commitment_cost  = commitment_cost   # weight for commitment loss (β in VQ-VAE)
        self.lambda_eik       = lambda_eik
        self.lambda_prior     = lambda_prior      # optional weight for prior CE
        self.lambda_far       = lambda_far        # weight for far-field SDF regularization

    def compute_sdf_loss(self, sdf_pred, sdf_gt):
        """
        Surface-weighted truncated SDF loss.
        Clamps sdf_pred to avoid NaNs from unstable model outputs (e.g. early training, AMP).

        Points near the zero-crossing define the mesh surface and account for
        only ~3% of samples, but are the only region where the sign changes.
        Without upweighting, the model ignores them and predicts the mean SDF
        (-0.021) everywhere, producing a constant-negative volume.

        Weight scheme:
          - |sdf_gt| < tau/2  (near-surface):          weight += 15
          - sdf_gt   < 0      (interior, rare ~4 %):   weight += 25
          - wrong sign (gt<0, pred>0):                  weight += 20
        """
        # Replace NaN/Inf from unstable forward (AMP, bad init) so loss stays finite
        sdf_pred_safe = torch.nan_to_num(sdf_pred, nan=0.0, posinf=self.tau, neginf=-self.tau)
        pred_clamped = torch.clamp(sdf_pred_safe, -self.tau, self.tau)
        gt_clamped   = torch.clamp(sdf_gt,        -self.tau, self.tau)

        # Per-point Huber loss (reduction='none' so we can apply weights).
        per_point_loss = F.smooth_l1_loss(pred_clamped, gt_clamped, reduction='none')

        # Standard SDF convention: sdf < 0  → inside mesh,  sdf > 0 → outside.
        # Interior points account for only ~4 % of samples but define the solid
        # volume; without upweighting the model can ignore them entirely and
        # predict positive (outside) everywhere, collapsing the surface.
        #
        # Weight scheme:
        #   near-surface (|sdf_gt| < tau/2): +15  — controls where the zero-crossing sits
        #   interior     (sdf_gt   < 0):     +25  — rare class, needs upweighting
        #   wrong sign   (gt < 0, pred > 0): +20  — predicted outside when actually inside
        near_surface_mask = (torch.abs(gt_clamped) < (self.tau * 0.5)).float()
        interior_mask     = (sdf_gt < 0).float()
        wrong_sign_mask   = ((sdf_gt < 0) & (pred_clamped > 0)).float()
        weights = 1.0 + (15.0 * near_surface_mask) + (25.0 * interior_mask) + (20.0 * wrong_sign_mask)

        loss_sdf = (weights * per_point_loss).mean()
        return loss_sdf

    def compute_far_loss(self, sdf_pred, sdf_gt_raw):
        """
        Far-field regularization loss.

        The truncated surface loss is blind to every point with |sdf_gt| > τ: both
        the GT and the prediction get clamped to the same ±τ constant, so the
        gradient is exactly zero there.  Without any signal in that region the model
        learns arbitrary SDF values far from the surface, which Marching Cubes
        interprets as extra zero-crossings — the floating fragments visible in the
        reconstructed mesh.

        This loss provides a soft gradient signal for those far-field points.
        We cap the GT at ±3τ (instead of using the raw value) so that the handful
        of points that are very far from the surface (raw SDF ≈ 1.4 in the corner
        of the unit cube) don't dominate and destabilise training.

        L_far = mean over far-field points of SmoothL1(pred_clamped_3τ, gt_clamped_3τ)
        """
        far_mask = (torch.abs(sdf_gt_raw) > self.tau)
        if not far_mask.any():
            return torch.tensor(0.0, device=sdf_pred.device)

        cap = 3.0 * self.tau
        gt_far   = torch.clamp(sdf_gt_raw, -cap, cap)
        pred_far = torch.clamp(sdf_pred,   -cap, cap)

        per_point = F.smooth_l1_loss(pred_far, gt_far, reduction='none')
        return per_point[far_mask].mean()

    def compute_vq_loss(self, codebook_loss, commitment_loss):
        """
        VQ-VAE regularisation term.
          codebook_loss   = ||sg[z_e] - e_k||^2  — moves codebook entries toward encoder
          commitment_loss = ||z_e - sg[e_k]||^2  — moves encoder output toward codebook
        Both are already scalar tensors produced inside VectorQuantizer.
        """
        return self.lambda_codebook * codebook_loss + self.commitment_cost * commitment_loss

    def compute_prior_loss(self, prior_logits, target_indices):
        """
        Prior classification loss over codebook indices.
        """
        return F.cross_entropy(prior_logits, target_indices.long())

    def compute_eikonal_loss(self, sdf_pred, points):
        """
        Eikonal regularization to enforce a valid SDF geometry. [cite: 170]
        L_eik = E[(||grad_x SDF||_2 - 1)^2]
        """
        # Compute d(sdf_pred)/d(points). Requires points.requires_grad_(True) before forward.
        grad_outputs = torch.ones_like(sdf_pred, requires_grad=False, device=sdf_pred.device)

        gradients = torch.autograd.grad(
            outputs=sdf_pred,
            inputs=points,
            grad_outputs=grad_outputs,
            create_graph=True,  # Needed to backprop through the gradient norm.
            retain_graph=True,
            only_inputs=True
        )[0]

        # Sanitize raw gradients before norm: fp16 autocast can produce Inf/NaN in
        # higher-order gradient computations (create_graph=True), which propagates to
        # the loss value and makes loss_finite differ across DDP ranks → NCCL deadlock.
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=10.0, neginf=-10.0)

        # (||grad|| - 1)^2. Clamp grad_norm to avoid residual fp16 overflow.
        grad_norm = gradients.norm(2, dim=-1)
        grad_norm = torch.clamp(grad_norm, 0.0, 10.0)
        eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm))
        return eikonal_loss

    def forward(self, sdf_pred, sdf_gt, codebook_loss, commitment_loss, points,
                prior_logits=None, target_indices=None):
        """
        Total loss: L = lambda_sdf*L_sdf + L_vq + lambda_eik*L_eik + lambda_far*L_far (+ optional lambda_prior*L_prior).
        """
        codebook_loss = torch.nan_to_num(codebook_loss, nan=0.0, posinf=1.0, neginf=0.0)
        commitment_loss = torch.nan_to_num(commitment_loss, nan=0.0, posinf=1.0, neginf=0.0)
        sdf_pred_safe = torch.nan_to_num(sdf_pred, nan=0.0, posinf=self.tau, neginf=-self.tau)
        l_sdf = self.compute_sdf_loss(sdf_pred, sdf_gt)
        l_vq  = self.compute_vq_loss(codebook_loss, commitment_loss)
        l_eik = self.compute_eikonal_loss(sdf_pred_safe, points)

        l_prior = torch.tensor(0.0, device=sdf_pred.device)
        if prior_logits is not None and target_indices is not None:
            l_prior = self.compute_prior_loss(prior_logits, target_indices)

        l_far = self.compute_far_loss(sdf_pred_safe, sdf_gt)

        total_loss = (self.lambda_sdf   * l_sdf
                      + l_vq
                      + self.lambda_eik   * l_eik
                      + self.lambda_prior * l_prior
                      + self.lambda_far   * l_far)

        return total_loss, {
            "loss_sdf":   l_sdf.item(),
            "loss_vq":    l_vq.item(),
            "loss_eik":   l_eik.item(),
            "loss_prior": l_prior.item(),
            "loss_far":   l_far.item(),
        }
