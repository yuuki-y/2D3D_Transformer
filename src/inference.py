import argparse
import torch
import nibabel as nib
import numpy as np

from model import XrayTo3D

def inference(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {args.model_path}")
    # The model parameters must match the ones used during training
    model = XrayTo3D(
        image_size=32,
        patch_size=4,
        enc_dim=512,
        enc_depth=6,
        enc_heads=8,
        enc_mlp_dim=1024,
        output_shape=(1, 32, 32, 32)
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Input Images
    print(f"Loading frontal image from: {args.frontal_image_path}")
    frontal_img = torch.load(args.frontal_image_path, map_location=device)

    print(f"Loading lateral image from: {args.lateral_image_path}")
    lateral_img = torch.load(args.lateral_image_path, map_location=device)

    # Preprocess (add batch dimension)
    frontal_img = frontal_img.unsqueeze(0)
    lateral_img = lateral_img.unsqueeze(0)

    # Perform Inference
    print("Running inference...")
    with torch.no_grad():
        recon_volume = model(frontal_img, lateral_img)
    print("Inference complete.")

    # Post-process
    # Remove batch and channel dimensions, and move to CPU
    recon_volume = recon_volume.squeeze(0).squeeze(0).cpu().numpy()

    # Save as NIfTI
    # Use identity affine matrix as a default
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(recon_volume, affine)

    print(f"Saving output to {args.output_path}")
    nib.save(nifti_img, args.output_path)
    print("Output saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with XrayTo3D model.")

    parser.add_argument('--frontal_image_path', type=str, required=True, help='Path to the frontal .pt image.')
    parser.add_argument('--lateral_image_path', type=str, required=True, help='Path to the lateral .pt image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--output_path', type=str, default='output.nii.gz', help='Path to save the output NIfTI file.')

    args = parser.parse_args()
    inference(args)
