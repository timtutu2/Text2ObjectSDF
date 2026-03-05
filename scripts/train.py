import os
import sys
import math
import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Ensure local repo root takes precedence over image-level PYTHONPATH entries (e.g. /app).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import Text2ObjectDataset
from src.models.network import Text2ObjectNetwork
from src.training.loss import Text2ObjectLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Text2ObjectSDF (single- or multi-GPU via torchrun).")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["stage1", "stage2"],
        help="Training stage. If omitted, uses training.stage in config (default: stage1).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume/load.",
    )
    parser.add_argument(
        "--reset_optim",
        action="store_true",
        help="Load model weights from --resume but reset optimizer/scheduler/epoch.",
    )
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_distributed = world_size > 1
    rank = 0
    local_rank = 0

    if use_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA GPUs, but CUDA is not available.")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "device": device,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "use_distributed": use_distributed,
        "is_main_process": rank == 0,
    }


def normalize_training_config(train_cfg):
    train_cfg["grad_accum_steps"] = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    return train_cfg


def all_reduce_mean(value, world_size, use_distributed):
    if not use_distributed:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def configure_stage(model, stage):
    # CLIP stays frozen in both stages.
    set_requires_grad(model.semantic_encoder, False)

    if stage == "stage1":
        # Stage 1: shape VQ + decoder only.
        set_requires_grad(model.vq_encoder, True)
        set_requires_grad(model.text_prior, False)
        set_requires_grad(model.spatial_encoder, True)
        set_requires_grad(model.decoder_layers, True)
        set_requires_grad(model.output_layer, True)
        return

    if stage == "stage2":
        # Stage 2: freeze shape VQ and decoder, train text prior only.
        # Note: leaving decoder trainable with prior-only loss would create unused params.
        set_requires_grad(model.vq_encoder, False)
        set_requires_grad(model.spatial_encoder, False)
        set_requires_grad(model.decoder_layers, False)
        set_requires_grad(model.output_layer, False)
        set_requires_grad(model.text_prior, True)
        return

    raise ValueError(f"Unsupported stage: {stage}")


def build_scheduler(optimizer, train_cfg, num_epochs, stage, is_main_process):
    warmup_epochs = int(train_cfg.get(f"warmup_epochs_{stage}", train_cfg.get("warmup_epochs", 50)))
    stable_epochs = int(train_cfg.get(f"stable_epochs_{stage}", train_cfg.get("stable_epochs", 600)))
    decay_epochs = max(num_epochs - warmup_epochs - stable_epochs, 1)
    base_lr = float(train_cfg["learning_rate"])
    eta_min = 1e-6

    if is_main_process:
        print(
            f"LR schedule ({stage}) — Warmup: {warmup_epochs} | "
            f"Stable: {stable_epochs} | Decay: {decay_epochs} epochs"
        )

    def _wsd_lambda(epoch):
        if epoch < warmup_epochs:
            return 1e-4 + (1.0 - 1e-4) * epoch / max(warmup_epochs, 1)
        if epoch < warmup_epochs + stable_epochs:
            return 1.0
        progress = (epoch - warmup_epochs - stable_epochs) / decay_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return eta_min / base_lr + (1.0 - eta_min / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_wsd_lambda)


def main():
    args = parse_args()
    ddp = init_distributed()
    device = ddp["device"]
    rank = ddp["rank"]
    local_rank = ddp["local_rank"]
    world_size = ddp["world_size"]
    use_distributed = ddp["use_distributed"]
    is_main_process = ddp["is_main_process"]

    if is_main_process:
        print(f"Using device: {device} | world_size={world_size}")

    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_cfg = normalize_training_config(config["training"])
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    log_cfg = config.get("logging", {})
    version_cfg = config.get("version", {})

    stage = (args.stage or train_cfg.get("stage", "stage1")).lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Invalid training stage: {stage}")

    num_epochs = int(train_cfg.get(f"num_epochs_{stage}", train_cfg.get("num_epochs", 1)))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    checkpoints_dir = PROJECT_ROOT / "checkpoints" / version_cfg.get("name", "default")
    if is_main_process:
        os.makedirs(checkpoints_dir, exist_ok=True)

    log_dir = log_cfg.get("log_dir", "/mnt/tim/text2objectsdf/logs")
    run_name = f"{config.get('experiment_name', 'text2object')}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = os.path.join(log_dir, run_name)
    if is_main_process:
        os.makedirs(tb_log_dir, exist_ok=True)
        print(f"TensorBoard logs: {tb_log_dir}")

    use_wandb = is_main_process and log_cfg.get("wandb_enabled", False) and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=log_cfg.get("wandb_project", "text2object-sdf"),
            name=run_name,
            config={
                "experiment_name": config.get("experiment_name"),
                "stage": stage,
                "training": train_cfg,
                "model": model_cfg,
                "loss": loss_cfg,
            },
        )
        print("Weights & Biases logging enabled.")
    elif is_main_process and log_cfg.get("wandb_enabled", False) and not _WANDB_AVAILABLE:
        print("wandb_enabled is true but 'wandb' not installed; skipping W&B. pip install wandb")

    dataset = Text2ObjectDataset(
        processed_dir1="/mnt/tim/data/ShapeNetCore/voxel_256_filter_div_128_solid_2",
        processed_dir2="/mnt/tim/data/ShapeNetCore/03001627_sdf",
        captions_file=str(PROJECT_ROOT / "src" / "data" / "captions_clip77.json"),
        num_points_per_batch=train_cfg["points_per_batch"],
        max_models=10000,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    if is_main_process:
        print("Loading Core Network (Text2ObjectNetwork)...")
    sdf_decoder = Text2ObjectNetwork(
        text_embed_dim=model_cfg["text_embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_embeddings=model_cfg.get("num_embeddings", 128),
        num_tokens=model_cfg.get("num_tokens", 8),
        hashgrid=model_cfg.get("hashgrid"),
    ).to(device)

    # IMPORTANT: freeze/unfreeze parameters before wrapping with DDP.
    # DDP expects a stable set of trainable params; changing requires_grad
    # after DDP construction can trigger "Expected to have finished reduction"
    # when some params stop receiving gradients (e.g., stage-specific branches).
    configure_stage(sdf_decoder, stage)

    model_without_ddp = sdf_decoder
    if use_distributed:
        sdf_decoder = torch.nn.parallel.DistributedDataParallel(
            sdf_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )
        model_without_ddp = sdf_decoder.module

    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters found for {stage}.")

    criterion = Text2ObjectLoss(
        truncation_dist=loss_cfg["truncation_dist"],
        lambda_sdf=loss_cfg.get("lambda_sdf", 1.0),
        lambda_codebook=loss_cfg.get("lambda_codebook", 1.0),
        commitment_cost=loss_cfg.get("commitment_cost", 0.25),
        lambda_eik=loss_cfg["lambda_eik"],
        lambda_prior=loss_cfg.get("lambda_prior", 1.0),
        lambda_far=loss_cfg.get("lambda_far", 0.1),
    ).to(device)

    optimizer = torch.optim.Adam(trainable_params, lr=train_cfg["learning_rate"])
    scheduler = build_scheduler(optimizer, train_cfg, num_epochs, stage, is_main_process)

    start_epoch = 0
    global_step = 0
    resume_path = args.resume.expanduser() if args.resume is not None else None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_without_ddp.load_state_dict(state_dict, strict=True)

        checkpoint_stage = checkpoint.get("training_stage", "unknown")
        should_restore_optim = (not args.reset_optim) and (checkpoint_stage == stage)

        if should_restore_optim:
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
            global_step = int(checkpoint.get("global_step", start_epoch * steps_per_epoch))

        if is_main_process:
            print(
                f"Loaded: {resume_path} | checkpoint_stage={checkpoint_stage} | "
                f"restore_optim={should_restore_optim} | start_epoch={start_epoch} | global_step={global_step}"
            )

    if is_main_process:
        print(
            f"Starting {stage} training loop... "
            f"Total Epochs: {num_epochs}, Per-GPU Batch Size: {train_cfg['batch_size']}, "
            f"Grad Accum: {grad_accum_steps}"
        )

    accumulated_finite = False
    loss = torch.tensor(0.0, device=device)

    for epoch in range(start_epoch, num_epochs):
        sdf_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, (points, sdf_gt, prompts) in enumerate(dataloader):
            is_update_step = (
                ((batch_idx + 1) % grad_accum_steps == 0)
                or ((batch_idx + 1) == len(dataloader))
            )
            sync_context = (
                sdf_decoder.no_sync()
                if use_distributed and not is_update_step
                else nullcontext()
            )

            points = points.to(device, non_blocking=True)
            sdf_gt = sdf_gt.to(device, non_blocking=True)
            if stage == "stage1":
                points.requires_grad_(True)
            else:
                points.requires_grad_(False)

            loss_finite = False
            with sync_context:
                if stage == "stage1":
                    sdf_pred, codebook_loss, commitment_loss, _ = sdf_decoder(
                        points, s_gt=sdf_gt, mode="stage1"
                    )
                    loss_sdf = criterion.compute_sdf_loss(sdf_pred, sdf_gt)
                    loss_vq = criterion.lambda_codebook * (
                        codebook_loss + criterion.commitment_cost * commitment_loss
                    )
                    sdf_pred_safe = torch.nan_to_num(
                        sdf_pred,
                        nan=0.0,
                        posinf=criterion.tau,
                        neginf=-criterion.tau,
                    )
                    loss_eik = criterion.compute_eikonal_loss(sdf_pred_safe, points)
                    loss = loss_sdf + loss_vq + (criterion.lambda_eik * loss_eik)
                    loss_dict = {
                        "loss_sdf": loss_sdf.item(),
                        "loss_vq": loss_vq.item(),
                        "loss_eik": loss_eik.item(),
                        "loss_prior": 0.0,
                        "loss_far": 0.0,
                    }
                else:
                    prior_logits, indices_gt = sdf_decoder(
                        points, prompts=prompts, s_gt=sdf_gt, mode="stage2"
                    )
                    # prior_logits: (B, T, K), indices_gt: (B, T)
                    bsz, num_tokens, num_embeddings = prior_logits.shape
                    per_token_ce = F.cross_entropy(
                        prior_logits.reshape(bsz * num_tokens, num_embeddings),  # (B*T, K)
                        indices_gt.long().reshape(bsz * num_tokens),             # (B*T,)
                        reduction="none",
                    ).view(bsz, num_tokens)                                       # (B, T)
                    prior_loss = per_token_ce.sum(dim=1).mean()                   # mean_B sum_T CE
                    loss = criterion.lambda_prior * prior_loss
                    loss_dict = {
                        "loss_sdf": 0.0,
                        "loss_vq": 0.0,
                        "loss_eik": 0.0,
                        "loss_prior": prior_loss.item(),
                        "loss_far": 0.0,
                    }

                loss_finite = torch.isfinite(loss).all().item()

                if is_update_step and use_distributed:
                    finite_flag = torch.tensor(float(loss_finite), device=device)
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                    loss_finite = finite_flag.item() > 0.5

                if loss_finite:
                    accumulated_finite = True
                    scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()

            if is_update_step:
                if not accumulated_finite:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                accumulated_finite = False

                reduced_total = all_reduce_mean(loss.detach(), world_size, use_distributed)
                reduced_sdf = all_reduce_mean(torch.tensor(loss_dict["loss_sdf"], device=device), world_size, use_distributed)
                reduced_vq = all_reduce_mean(torch.tensor(loss_dict["loss_vq"], device=device), world_size, use_distributed)
                reduced_eik = all_reduce_mean(torch.tensor(loss_dict["loss_eik"], device=device), world_size, use_distributed)
                reduced_prior = all_reduce_mean(torch.tensor(loss_dict["loss_prior"], device=device), world_size, use_distributed)
                reduced_far = all_reduce_mean(torch.tensor(loss_dict["loss_far"], device=device), world_size, use_distributed)

                if use_wandb:
                    log_data = {
                        "Loss/Total": reduced_total.item(),
                        "Loss/Prior": reduced_prior.item(),
                        "LR": scheduler.get_last_lr()[0],
                    }
                    if stage == "stage1":
                        log_data.update({
                            "Loss/SDF": reduced_sdf.item(),
                            "Loss/VQ": reduced_vq.item(),
                            "Loss/Eikonal": reduced_eik.item(),
                            "Loss/FarField": reduced_far.item(),
                        })
                    wandb.log(log_data, step=global_step)

                total_val = reduced_total.item()
                if is_main_process:
                    if total_val != total_val:
                        print(
                            f"[WARNING] NaN loss at step {global_step}. "
                            f"SDF={loss_dict['loss_sdf']:.4f} VQ={loss_dict['loss_vq']:.4f} "
                            f"Eik={loss_dict['loss_eik']:.4f} Prior={loss_dict['loss_prior']:.4f}."
                        )
                    elif global_step % 50 == 0:
                        print(
                            f"{stage} Epoch [{epoch+1}/{num_epochs}] "
                            f"Step [{global_step}]: Total Loss: {total_val:.4f}"
                        )

                global_step += 1

        if is_main_process and (epoch + 1) % train_cfg["save_interval"] == 0:
            checkpoint_path = checkpoints_dir / f"{stage}_model_epoch_{epoch+1}.pth"
            torch.save({
                "training_stage": stage,
                "epoch": epoch + 1,
                "model_state_dict": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "loss": loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

        scheduler.step()

    if is_main_process:
        final_model_path = checkpoints_dir / f"{stage}_model_final.pth"
        torch.save({
            "training_stage": stage,
            "epoch": num_epochs,
            "model_state_dict": model_without_ddp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "loss": loss.item(),
        }, final_model_path)
        print(f"Training finished! Final model saved to: {final_model_path}")

    if use_wandb:
        wandb.finish()

    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
