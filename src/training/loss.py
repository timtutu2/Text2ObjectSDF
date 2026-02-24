import torch
import torch.nn as nn
import torch.nn.functional as F

class Text2ObjectLoss(nn.Module):
    """
    Combined loss function for the VQ-VAE SDF network.
    L = L_sdf + lambda_cb * L_codebook + lambda_cm * L_commitment + lambda_eik * L_eik
    """
    def __init__(self, truncation_dist=0.1, lambda_codebook=1.0,
                 commitment_cost=0.25, lambda_eik=0.1):
        super().__init__()
        self.tau              = truncation_dist
        self.lambda_codebook  = lambda_codebook   # weight for codebook loss
        self.commitment_cost  = commitment_cost   # weight for commitment loss (β in VQ-VAE)
        self.lambda_eik       = lambda_eik

    def compute_sdf_loss(self, sdf_pred, sdf_gt):
        """
        Surface-weighted truncated SDF loss.

        Points near the zero-crossing define the mesh surface and account for
        only ~3% of samples, but are the only region where the sign changes.
        Without upweighting, the model ignores them and predicts the mean SDF
        (-0.021) everywhere, producing a constant-negative volume.

        Weight scheme:
          - |sdf_gt| < tau/2  (near-surface): weight = 10.0
          - otherwise (far field):             weight = 1.0
        """
        pred_clamped = torch.clamp(sdf_pred, -self.tau, self.tau)
        gt_clamped   = torch.clamp(sdf_gt,   -self.tau, self.tau)

        # Per-point Huber loss (reduction='none' so we can apply weights).
        per_point_loss = F.smooth_l1_loss(pred_clamped, gt_clamped, reduction='none')

        # Up-weight the near-surface band (|sdf_gt| < tau/2).
        near_surface_mask = (torch.abs(gt_clamped) < (self.tau * 0.5)).float()
        inside_mask = (sdf_gt > 0).float()
        wrong_sign_mask = ((sdf_gt > 0) & (sdf_pred < 0)).float()
        weights = 1.0 + (15.0 * near_surface_mask) + (25.0 * inside_mask) + (20.0 * wrong_sign_mask)

        loss_sdf = (weights * per_point_loss).mean()
        return loss_sdf

    def compute_vq_loss(self, codebook_loss, commitment_loss):
        """
        VQ-VAE regularisation term.
          codebook_loss   = ||sg[z_e] - e_k||^2  — moves codebook entries toward encoder
          commitment_loss = ||z_e - sg[e_k]||^2  — moves encoder output toward codebook
        Both are already scalar tensors produced inside VectorQuantizer.
        """
        return self.lambda_codebook * codebook_loss + self.commitment_cost * commitment_loss

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

        # (||grad|| - 1)^2
        grad_norm = gradients.norm(2, dim=-1)
        eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm))
        return eikonal_loss

    def forward(self, sdf_pred, sdf_gt, codebook_loss, commitment_loss, points):
        """
        Total loss: L = L_sdf + L_vq + lambda_eik * L_eik.
        """
        l_sdf = self.compute_sdf_loss(sdf_pred, sdf_gt)
        l_vq  = self.compute_vq_loss(codebook_loss, commitment_loss)
        l_eik = self.compute_eikonal_loss(sdf_pred, points)

        total_loss = l_sdf + l_vq + (self.lambda_eik * l_eik)

        return total_loss, {
            "loss_sdf": l_sdf.item(),
            "loss_vq":  l_vq.item(),
            "loss_eik": l_eik.item()
        }