import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import XrayTo3D
from dataset import Xray3DDataset
from train import validate

def get_default_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--frontal_dir', type=str, default='data/frontal')
    parser.add_argument('--lateral_dir', type=str, default='data/lateral')
    parser.add_argument('--gt_dir', type=str, default='data/ground_truth')

    # Training params
    parser.add_argument('--epochs', type=int, default=1) # 1 epoch for each trial is fast
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Model dimensions
    # Using smaller sizes for dummy data compatibility during tuning
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--volume_size', type=int, default=32)

    # Model architecture params
    parser.add_argument('--enc_dim', type=int, default=512)
    parser.add_argument('--enc_depth', type=int, default=6)
    parser.add_argument('--enc_heads', type=int, default=8)
    parser.add_argument('--enc_mlp_dim', type=int, default=1024)

    # Other args needed by validate function
    parser.add_argument('--amp', action='store_true', default=False)

    # Optuna runs on dummy data, so we parse with an empty list
    args, _ = parser.parse_known_args()
    return args


def objective(trial):
    """
    Optuna objective function.
    """
    args = get_default_args()

    # Suggest hyperparameters to tune
    args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    args.enc_depth = trial.suggest_int('enc_depth', 2, 4) # smaller range for faster trials
    args.enc_heads = trial.suggest_categorical('enc_heads', [2, 4, 8])

    print(f"\n--- Trial {trial.number} ---")
    print(f"  Params: lr={args.lr:.6f}, enc_depth={args.enc_depth}, enc_heads={args.enc_heads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We use the small dummy dataset for tuning
    full_dataset = Xray3DDataset(
        frontal_dir=args.frontal_dir, lateral_dir=args.lateral_dir, gt_dir=args.gt_dir
    )
    val_size = len(full_dataset) // 5
    if val_size == 0 and len(full_dataset) > 1: val_size = 1
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model with tuned parameters
    model = XrayTo3D(
        image_size=args.image_size,
        patch_size=args.patch_size,
        volume_size=args.volume_size,
        enc_dim=args.enc_dim,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        enc_mlp_dim=args.enc_mlp_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Simplified training loop for one epoch
    model.train()
    for batch in train_loader:
        frontal_imgs, lateral_imgs, gt_volumes = batch['frontal'].to(device), batch['lateral'].to(device), batch['ground_truth'].to(device)
        optimizer.zero_grad()
        recon_volumes = model(frontal_imgs, lateral_imgs)
        loss = criterion(recon_volumes, gt_volumes)
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = validate(model, val_loader, criterion, device, args)
    print(f"  Validation Loss for trial {trial.number}: {val_loss:.4f}")

    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) # n_trials can be increased for a real search

    print("\n--- Optuna Study Best Trial ---")
    trial = study.best_trial
    print(f"  Value (min val loss): {trial.value:.4f}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
