import argparse
import optuna
from train import train, validate
from model import XrayTo3D
from dataset import Xray3DDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# We need to get the default args from train.py to modify them
# A simple way is to create a new parser here, but it's better to import
# However, to avoid running the train() function when importing, we ensure
# the main execution block in train.py is guarded by if __name__ == '__main__'
# which is already done.
# For simplicity here, let's redefine the parser and get the defaults.

def get_default_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--frontal_dir', type=str, default='data/frontal')
    parser.add_argument('--lateral_dir', type=str, default='data/lateral')
    parser.add_argument('--gt_dir', type=str, default='data/ground_truth')
    parser.add_argument('--save_dir', type=str, default='checkpoints_tune') # Use a different dir for tuning
    # Training params
    parser.add_argument('--epochs', type=int, default=3) # Fewer epochs for faster tuning trials
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10) # Don't save models during tuning
    # AMP
    parser.add_argument('--amp', action='store_true', default=True)
    # WandB - disabled during tuning to avoid clutter
    parser.add_argument('--use_wandb', action='store_false')
    parser.add_argument('--wandb_project', type=str, default='xray-to-3d-tuning')
    parser.add_argument('--wandb_entity', type=str, default=None)

    # Model params to tune
    parser.add_argument('--enc_depth', type=int, default=6)
    parser.add_argument('--enc_heads', type=int, default=8)
    parser.add_argument('--enc_dim', type=int, default=512)

    return parser.parse_args([])


def objective(trial):
    """
    Optuna objective function.
    This function is called for each trial to train a model with a set of hyperparameters.
    """
    args = get_default_args()

    # Suggest hyperparameters
    args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    args.enc_depth = trial.suggest_int('enc_depth', 2, 6)
    # You can add more hyperparameters to tune here, e.g.:
    # args.enc_heads = trial.suggest_categorical('enc_heads', [4, 8])
    # args.batch_size = trial.suggest_categorical('batch_size', [1, 2])

    print(f"--- Trial {trial.number} ---")
    print(f"  Params: lr={args.lr:.6f}, enc_depth={args.enc_depth}")

    # The train function needs to be refactored to accept model params
    # Let's adjust train.py to accept them or create a new train function here.
    # For now, let's create a simplified training loop inside the objective.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = Xray3DDataset(
        frontal_dir=args.frontal_dir, lateral_dir=args.lateral_dir, gt_dir=args.gt_dir
    )
    val_size = len(full_dataset) // 5
    if val_size == 0 and len(full_dataset) > 1: val_size = 1
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = XrayTo3D(
        image_size=32, patch_size=4, enc_dim=args.enc_dim, enc_depth=args.enc_depth,
        enc_heads=args.enc_heads, enc_mlp_dim=1024, output_shape=(1, 32, 32, 32)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Simplified training loop for one epoch
    model.train()
    for batch in train_loader:
        frontal_imgs = batch['frontal'].to(device)
        lateral_imgs = batch['lateral'].to(device)
        gt_volumes = batch['ground_truth'].to(device)
        optimizer.zero_grad()
        recon_volumes = model(frontal_imgs, lateral_imgs)
        loss = criterion(recon_volumes, gt_volumes)
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = validate(model, val_loader, criterion, device, args)
    print(f"  Validation Loss: {val_loss:.4f}")

    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    # Using a small number of trials for verification
    study.optimize(objective, n_trials=3)

    print("\n--- Optuna Study Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
