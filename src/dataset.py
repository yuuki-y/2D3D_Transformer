import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class Xray3DDataset(Dataset):
    """
    Dataset for loading 2D X-ray images (.pt) and 3D ground truth volumes (.nii.gz).
    """
    def __init__(self, frontal_dir, lateral_dir, gt_dir, transform=None):
        """
        Args:
            frontal_dir (string): Directory with all the frontal images.
            lateral_dir (string): Directory with all the lateral images.
            gt_dir (string): Directory with all the ground truth volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frontal_dir = frontal_dir
        self.lateral_dir = lateral_dir
        self.gt_dir = gt_dir
        self.transform = transform

        # Assuming file names are consistent across directories, e.g., '001.pt', '001.nii.gz'
        self.frontal_files = sorted([f for f in os.listdir(frontal_dir) if f.endswith('.pt')])
        self.lateral_files = sorted([f for f in os.listdir(lateral_dir) if f.endswith('.pt')])
        self.gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])

        # Basic check to ensure file counts match
        if not (len(self.frontal_files) == len(self.lateral_files) == len(self.gt_files)):
            raise ValueError("Number of files in frontal, lateral, and ground truth directories must be the same.")

    def __len__(self):
        return len(self.frontal_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construct file paths
        base_name = os.path.splitext(self.frontal_files[idx])[0]
        frontal_img_path = os.path.join(self.frontal_dir, f"{base_name}.pt")
        lateral_img_path = os.path.join(self.lateral_dir, f"{base_name}.pt")
        gt_volume_path = os.path.join(self.gt_dir, f"{base_name}.nii.gz")

        # Load data
        frontal_img = torch.load(frontal_img_path)
        lateral_img = torch.load(lateral_img_path)

        # Load NIfTI file and get data as a numpy array
        gt_nifti = nib.load(gt_volume_path)
        gt_volume = torch.from_numpy(gt_nifti.get_fdata(dtype=np.float32))

        # Add a channel dimension to the ground truth volume to make it (1, D, H, W)
        gt_volume = gt_volume.unsqueeze(0)

        sample = {'frontal': frontal_img, 'lateral': lateral_img, 'ground_truth': gt_volume}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    # Example of how to use the dataset
    # This block will only be used for verification

    # Create a dummy dataset instance
    frontal_path = 'data/frontal'
    lateral_path = 'data/lateral'
    gt_path = 'data/ground_truth'

    dataset = Xray3DDataset(frontal_dir=frontal_path, lateral_dir=lateral_path, gt_dir=gt_path)

    print(f"Dataset size: {len(dataset)}")

    # Get one sample
    sample = dataset[0]
    frontal_img = sample['frontal']
    lateral_img = sample['lateral']
    gt_volume = sample['ground_truth']

    print(f"Frontal image shape: {frontal_img.shape}")
    print(f"Lateral image shape: {lateral_img.shape}")
    print(f"Ground truth volume shape: {gt_volume.shape}")

    # Check tensor types
    assert isinstance(frontal_img, torch.Tensor)
    assert isinstance(lateral_img, torch.Tensor)
    assert isinstance(gt_volume, torch.Tensor)

    print("\nDataset implementation seems correct.")
