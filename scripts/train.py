import os
import yaml
import torch
from torch.utils.data import DataLoader
from datetime import datetime

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src.data.dataset import Text2ObjectDataset
from src.models.network import Text2ObjectNetwork
from src.training.loss import Text2ObjectLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration from YAML file
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configurations
    train_cfg = config['training']
    model_cfg = config['model']
    loss_cfg = config['loss']
    log_cfg = config.get('logging', {})

    # Create output directories
    checkpoints_dir = "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # TensorBoard: log_dir with timestamped run subdir
    log_dir = log_cfg.get('log_dir', '/mnt/tim/text2objectsdf/logs')
    run_name = f"{config.get('experiment_name', 'text2object')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_log_dir = os.path.join(log_dir, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    print(f"TensorBoard logs: {tb_log_dir}")

    # Weights & Biases (optional; uses WANDB_API_KEY from env, e.g. K8s secret)
    use_wandb = log_cfg.get('wandb_enabled', False) and _WANDB_AVAILABLE
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
    elif log_cfg.get('wandb_enabled', False) and not _WANDB_AVAILABLE:
        print("wandb_enabled is true but 'wandb' not installed; skipping W&B. pip install wandb")

    # Initialize dataset and dataloader
    dataset = Text2ObjectDataset(
        processed_dir1="/mnt/tim/data/ShapeNetCore/04379243_sdf", 
        processed_dir2="/mnt/tim/data/ShapeNetCore/03001627_sdf",
        captions_file="src/data/captions.json",
        num_points_per_batch=train_cfg['points_per_batch']
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        drop_last=True,
        num_workers=train_cfg['num_workers']
    )

    # Initialize network and loss function
    print("Loading Core Network (Text2ObjectNetwork)...")
    sdf_decoder = Text2ObjectNetwork(
        text_embed_dim=model_cfg['text_embed_dim'], 
        latent_dim=model_cfg['latent_dim'], 
        hidden_dim=model_cfg['hidden_dim'],
        num_layers=model_cfg['num_layers'],
        num_embeddings=model_cfg.get('num_embeddings', 512),
        hashgrid=model_cfg.get('hashgrid'),
    ).to(device)
    
    criterion = Text2ObjectLoss(
        truncation_dist=loss_cfg['truncation_dist'],
        lambda_codebook=loss_cfg.get('lambda_codebook', 1.0),
        commitment_cost=loss_cfg.get('commitment_cost', 0.25),
        lambda_eik=loss_cfg['lambda_eik']
    ).to(device)
    
    optimizer = torch.optim.Adam(sdf_decoder.parameters(), lr=train_cfg['learning_rate'])

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

    print(f"Starting training loop... Total Epochs: {train_cfg['num_epochs']}, Batch Size: {train_cfg['batch_size']}")
    global_step = 0

    for epoch in range(train_cfg['num_epochs']): 
        sdf_decoder.train()
        
        for batch_idx, (points, sdf_gt, prompts) in enumerate(dataloader):
            points = points.to(device)   
            sdf_gt = sdf_gt.to(device)   
            
            # Enable gradient tracking for Eikonal loss
            points.requires_grad_(True)
            optimizer.zero_grad()

            # Forward pass
            sdf_pred, codebook_loss, commitment_loss = sdf_decoder(points, prompts, s_gt=sdf_gt)

            # Loss computation & backpropagation
            loss, loss_dict = criterion(sdf_pred, sdf_gt, codebook_loss, commitment_loss, points)
            loss.backward()

            # Gradient clipping: prevents exploding gradients during early training
            # when the newly-corrected output_layer init produces larger gradient magnitudes.
            torch.nn.utils.clip_grad_norm_(sdf_decoder.parameters(), max_norm=1.0)

            optimizer.step()

            # Weights & Biases (same metrics)
            if use_wandb:
                wandb.log({
                    'Loss/Total': loss.item(),
                    'Loss/SDF': loss_dict['loss_sdf'],
                    'Loss/VQ': loss_dict['loss_vq'],
                    'Loss/Eikonal': loss_dict['loss_eik'],
                    'LR': scheduler.get_last_lr()[0],
                }, step=global_step)

            if global_step % 50 == 0:
                print(f"Epoch [{epoch+1}/{train_cfg['num_epochs']}] Step [{global_step}]: Total Loss: {loss.item():.4f}")
            
            global_step += 1

        # Checkpointing
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch+1}.pth")
            
            checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': sdf_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

        scheduler.step()

    final_model_path = os.path.join(checkpoints_dir, "model_final.pth")
    torch.save({
        'epoch': train_cfg['num_epochs'],
        'model_state_dict': sdf_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if 'loss' in locals() else 0.0,
    }, final_model_path)
    print(f"Training finished! Final model saved to: {final_model_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()