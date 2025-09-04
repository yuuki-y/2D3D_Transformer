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
    # The model parameters must match the ones used for the saved checkpoint
    model = XrayTo3D(
        image_size=args.image_size,
        patch_size=args.patch_size,
        volume_size=args.volume_size,
        # The other params like enc_dim, depth, etc., must also match the trained model.
        # For simplicity, we assume they are the defaults. A more robust solution
        # would save model hyperparameters in the checkpoint file.
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
    recon_volume = recon_volume.squeeze(0).squeeze(0).cpu().numpy()

    # Save as NIfTI
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(recon_volume, affine)

    print(f"Saving output to {args.output_path}")
    nib.save(nifti_img, args.output_path)
    print("Output saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with XrayTo3D model.")

    # Required paths
    parser.add_argument('--frontal_image_path', type=str, required=True, help='Path to the frontal .pt image.')
    parser.add_argument('--lateral_image_path', type=str, required=True, help='Path to the lateral .pt image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('--output_path', type=str, default='output.nii.gz', help='Path to save the output NIfTI file.')

    # Model dimensions - should match the trained model
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input 2D images.')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for the ViT encoder.')
    parser.add_argument('--volume_size', type=int, default=256, help='Size of the output 3D volume.')

    args = parser.parse_args()
    inference(args)
