import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train Text2ObjectSDF.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the experiment config YAML.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-process batch size.")
    parser.add_argument("--points-per-batch", type=int, default=None, help="Override points sampled per object.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override optimizer learning rate.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint save interval.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation steps.")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", help="Enable fp16 mixed precision on CUDA.")
    amp_group.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
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


def override_training_config(train_cfg, args):
    overrides = {
        "batch_size": args.batch_size,
        "points_per_batch": args.points_per_batch,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "save_interval": args.save_interval,
        "grad_accum_steps": args.grad_accum_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            train_cfg[key] = value

    if args.amp:
        train_cfg["amp"] = True
    elif args.no_amp:
        train_cfg["amp"] = False

    train_cfg["grad_accum_steps"] = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    train_cfg["amp"] = bool(train_cfg.get("amp", False))
    return train_cfg


def reduce_mean(value, world_size, use_distributed):
    if not use_distributed:
        return value

    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


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

    # Load configuration from YAML file
    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configurations
    train_cfg = config['training']
    train_cfg = override_training_config(train_cfg, args)
    model_cfg = config['model']
    loss_cfg = config['loss']
    log_cfg = config.get('logging', {})

    # Create output directories
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # TensorBoard: log_dir with timestamped run subdir
    log_dir = log_cfg.get('log_dir', '/mnt/tim/text2objectsdf/logs')
    run_name = f"{config.get('experiment_name', 'text2object')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = os.path.join(log_dir, run_name)
    if is_main_process:
        os.makedirs(tb_log_dir, exist_ok=True)
        print(f"TensorBoard logs: {tb_log_dir}")

    # Weights & Biases (optional; uses WANDB_API_KEY from env, e.g. K8s secret)
    use_wandb = is_main_process and log_cfg.get('wandb_enabled', False) and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=log_cfg.get('wandb_project', 'text2object-sdf'),
            name=run_name,
            config={
                'experiment_name': config.get('experiment_name'),
                'training': train_cfg,
                'model': model_cfg,
                'loss': loss_cfg,
            },
        )
        print("Weights & Biases logging enabled.")
    elif is_main_process and log_cfg.get('wandb_enabled', False) and not _WANDB_AVAILABLE:
        print("wandb_enabled is true but 'wandb' not installed; skipping W&B. pip install wandb")

    # Initialize dataset and dataloader
    dataset = Text2ObjectDataset(
        processed_dir1="/mnt/tim/data/ShapeNetCore/04379243_sdf", 
        processed_dir2="/mnt/tim/data/ShapeNetCore/03001627_sdf",
        captions_file=str(PROJECT_ROOT / "src" / "data" / "captions.json"),
        num_points_per_batch=train_cfg['points_per_batch']
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_distributed else None
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train_cfg['num_workers'] > 0,
    )

    # Initialize network and loss function
    if is_main_process:
        print("Loading Core Network (Text2ObjectNetwork)...")
    sdf_decoder = Text2ObjectNetwork(
        text_embed_dim=model_cfg['text_embed_dim'], 
        latent_dim=model_cfg['latent_dim'], 
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_embeddings=model_cfg.get('num_embeddings', 512),
        hashgrid=model_cfg.get('hashgrid'),
    ).to(device)

    # DDP can hang/slow when output_layer (1, 256) grad has strides (1,1) vs bucket (256,1).
    # Force contiguous grad as soon as it is computed so all-reduce is fast.
    def _make_grad_contiguous(grad):
        return grad.contiguous() if grad is not None else None
    sdf_decoder.output_layer.weight.register_hook(_make_grad_contiguous)

    if use_distributed:
        sdf_decoder = torch.nn.parallel.DistributedDataParallel(
            sdf_decoder,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    model_without_ddp = sdf_decoder.module if use_distributed else sdf_decoder
    
    criterion = Text2ObjectLoss(
        truncation_dist=loss_cfg['truncation_dist'],
        lambda_codebook=loss_cfg.get('lambda_codebook', 1.0),
        commitment_cost=loss_cfg.get('commitment_cost', 0.25),
        lambda_eik=loss_cfg['lambda_eik']
    ).to(device)
    
    optimizer = torch.optim.Adam(sdf_decoder.parameters(), lr=train_cfg['learning_rate'])
    amp_enabled = device.type == "cuda" and train_cfg.get("amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    # Warmup → Stable → Decay (WSD) schedule.
    # Phase 1 – Warmup  (warmup_epochs):  LR ramps linearly from lr*1e-4 → lr.
    #   Prevents exploding gradients before the HashGrid features initialise.
    # Phase 2 – Stable  (stable_epochs):  LR stays constant at lr.
    #   Lets the model converge at full learning capacity.
    # Phase 3 – Decay   (remaining):      LR decays via cosine from lr → eta_min.
    #   Smoothly anneals into the final optimum without overshooting.
    warmup_epochs = train_cfg.get('warmup_epochs', 50)
    stable_epochs = train_cfg.get('stable_epochs', 600)
    decay_epochs  = max(train_cfg['num_epochs'] - warmup_epochs - stable_epochs, 1)

    if is_main_process:
        print(f"LR schedule — Warmup: {warmup_epochs} | Stable: {stable_epochs} | Decay: {decay_epochs} epochs")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,   # starts at learning_rate * 1e-4 ≈ ~0
        end_factor=1.0,      # ends at full learning_rate
        total_iters=warmup_epochs
    )
    stable_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,          # multiplicative factor = 1 → LR unchanged
        total_iters=stable_epochs
    )
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler],
        milestones=[warmup_epochs, warmup_epochs + stable_epochs]
    )

    if is_main_process:
        print(
            "Starting training loop... "
            f"Total Epochs: {train_cfg['num_epochs']}, "
            f"Per-GPU Batch Size: {train_cfg['batch_size']}, "
            f"Grad Accum: {grad_accum_steps}, AMP: {amp_enabled}"
        )
        if use_distributed:
            print("Note: First backward (Eikonal + DDP sync) can take 1–3 min; later steps are faster.")
    global_step = 0
    accumulated_finite = False  # True if at least one batch in current accum had finite loss

    for epoch in range(train_cfg['num_epochs']): 
        sdf_decoder.train()
        optimizer.zero_grad(set_to_none=True)
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        for batch_idx, (points, sdf_gt, prompts) in enumerate(dataloader):
            is_update_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
            sync_context = (
                sdf_decoder.no_sync()
                if use_distributed and not is_update_step
                else nullcontext()
            )

            points = points.to(device, non_blocking=True)
            sdf_gt = sdf_gt.to(device, non_blocking=True)
            
            # Enable gradient tracking for Eikonal loss
            points.requires_grad_(True)
            loss_finite = False
            with sync_context:
                with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.float16):
                    sdf_pred, codebook_loss, commitment_loss = sdf_decoder(points, prompts, s_gt=sdf_gt)

                    # Loss computation & backpropagation
                    loss, loss_dict = criterion(sdf_pred, sdf_gt, codebook_loss, commitment_loss, points)

                loss_finite = torch.isfinite(loss).all().item()
                if loss_finite:
                    accumulated_finite = True
                    scaled_loss = loss / grad_accum_steps
                    if amp_enabled:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

            if is_update_step:
                if not accumulated_finite:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # Make grads contiguous so DDP all-reduce doesn't hang/slow on stride mismatch
                    # (e.g. output_layer weight (1, 256) can produce non-contiguous grad)
                    for p in sdf_decoder.parameters():
                        if p.grad is not None and not p.grad.is_contiguous():
                            p.grad = p.grad.contiguous()

                    if amp_enabled:
                        scaler.unscale_(optimizer)

                    # Gradient clipping: prevents exploding gradients during early training
                    torch.nn.utils.clip_grad_norm_(sdf_decoder.parameters(), max_norm=1.0)

                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                accumulated_finite = False

                reduced_total = reduce_mean(loss.detach(), world_size, use_distributed)
                reduced_sdf = reduce_mean(torch.tensor(loss_dict['loss_sdf'], device=device), world_size, use_distributed)
                reduced_vq = reduce_mean(torch.tensor(loss_dict['loss_vq'], device=device), world_size, use_distributed)
                reduced_eik = reduce_mean(torch.tensor(loss_dict['loss_eik'], device=device), world_size, use_distributed)

                # Weights & Biases (same metrics)
                if use_wandb:
                    wandb.log({
                        'Loss/Total': reduced_total.item(),
                        'Loss/SDF': reduced_sdf.item(),
                        'Loss/VQ': reduced_vq.item(),
                        'Loss/Eikonal': reduced_eik.item(),
                        'LR': scheduler.get_last_lr()[0],
                    }, step=global_step)

                total_val = reduced_total.item()
                if is_main_process:
                    if not (total_val == total_val):  # NaN check
                        print(
                            f"[WARNING] NaN loss at step {global_step}. "
                            f"SDF={loss_dict['loss_sdf']} VQ={loss_dict['loss_vq']} Eik={loss_dict['loss_eik']}. "
                            "Try --no-amp to rule out fp16 overflow."
                        )
                    elif global_step % 50 == 0:
                        print(
                            f"Epoch [{epoch+1}/{train_cfg['num_epochs']}] "
                            f"Step [{global_step}]: Total Loss: {total_val:.4f}"
                        )

                global_step += 1

        # Checkpointing
        if is_main_process and (epoch + 1) % train_cfg['save_interval'] == 0:
            checkpoint_path = checkpoints_dir / f"model_epoch_{epoch+1}.pth"
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

        scheduler.step()

    if is_main_process:
        final_model_path = checkpoints_dir / "model_final.pth"
        torch.save({
            'epoch': train_cfg['num_epochs'],
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item() if 'loss' in locals() else 0.0,
        }, final_model_path)
        print(f"Training finished! Final model saved to: {final_model_path}")

    if use_wandb:
        wandb.finish()

    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
