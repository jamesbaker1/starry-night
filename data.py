import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class DepthDataset(Dataset):
    def __init__(self, image_paths, depth_paths, transform=None):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = plt.imread(self.image_paths[idx])
        depth = np.load(self.depth_paths[idx])

        if self.transform:
            image = self.transform(image)

        # Normalize depth values if necessary
        depth = torch.from_numpy(depth).float()
        return image, depth



def main():
    # Example paths - replace these with your actual data paths
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
    depth_paths = ['path/to/depth1.npy', 'path/to/depth2.npy']

    # Create the dataset
    dataset = DepthDataset(image_paths, depth_paths, transform=transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Example of iterating through the dataloader
    for batch_idx, (images, depths) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Image shape: {images.shape}")
        print(f"Depth shape: {depths.shape}")
        print("---")

        # Break after the first batch for this example
        if batch_idx == 0:
            break

if __name__ == "__main__":
    main()

# Add this line at the end of the file
__all__ = ['DepthDataset']
