import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb

from dataset import Xray3DDataset
from model import XrayTo3D

def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            frontal_imgs = batch['frontal'].to(device)
            lateral_imgs = batch['lateral'].to(device)
            gt_volumes = batch['ground_truth'].to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon_volumes = model(frontal_imgs, lateral_imgs)
                loss = criterion(recon_volumes, gt_volumes)
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    return total_loss / len(dataloader)

def train(args):
    # Setup WandB
    if args.use_wandb:
        # In a non-interactive environment, run in offline mode.
        # To run in online mode, set the WANDB_API_KEY environment variable.
        mode = "offline" if os.getenv("WANDB_API_KEY") is None else "online"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args, mode=mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # DataLoaders
    full_dataset = Xray3DDataset(
        frontal_dir=args.frontal_dir, lateral_dir=args.lateral_dir, gt_dir=args.gt_dir
    )
    val_size = len(full_dataset) // 5
    if val_size == 0 and len(full_dataset) > 1:
        val_size = 1
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # Model
    model = XrayTo3D(
        image_size=args.image_size,
        patch_size=args.patch_size,
        volume_size=args.volume_size,
        enc_dim=512,
        enc_depth=6,
        enc_heads=8,
        enc_mlp_dim=1024
    ).to(device)
    if args.use_wandb:
        wandb.watch(model, log='all')

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_train_loss = 0.0

        for i, batch in enumerate(pbar):
            frontal_imgs = batch['frontal'].to(device)
            lateral_imgs = batch['lateral'].to(device)
            gt_volumes = batch['ground_truth'].to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon_volumes = model(frontal_imgs, lateral_imgs)
                loss = criterion(recon_volumes, gt_volumes)
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            current_loss = loss.item() * args.accumulation_steps
            total_train_loss += current_loss
            pbar.set_postfix({'train_loss': f"{current_loss:.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device, args)

        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

        if (epoch + 1) % args.save_interval == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    if args.use_wandb:
        wandb.finish()

    print("Training finished.")
    return avg_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train XrayTo3D model.")

    # Paths
    parser.add_argument('--frontal_dir', type=str, default='data/frontal')
    parser.add_argument('--lateral_dir', type=str, default='data/lateral')
    parser.add_argument('--gt_dir', type=str, default='data/ground_truth')
    parser.add_argument('--save_dir', type=str, default='checkpoints')

    # Training params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)

    # AMP
    parser.add_argument('--amp', action='store_true')

    # WandB
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging.')
    parser.add_argument('--wandb_project', type=str, default='xray-to-3d', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity (username or team).')

    # Model dimensions
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input 2D images.')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for the ViT encoder.')
    parser.add_argument('--volume_size', type=int, default=256, help='Size of the output 3D volume.')

    args = parser.parse_args()
    train(args)
