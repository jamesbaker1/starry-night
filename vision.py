from data import DepthDataset as DepthDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbed, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [batch_size, embed_dim, H', W']
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class VisionTransformerDepth(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super(VisionTransformerDepth, self).__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        # Positional embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Prediction head
        self.depth_head = DepthDecoder(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x += self.pos_embed

        for block in self.blocks:
            x = block(x)

        depth = self.depth_head(x)
        return depth


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.transpose(0, 1)  # Required for nn.MultiheadAttention
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x = x.transpose(0, 1)
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class DepthDecoder(nn.Module):
    def __init__(self, embed_dim, img_size=224, patch_size=16):
        super(DepthDecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.size()
        h = w = int(np.sqrt(num_patches))
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, embed_dim, h, w)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return x.squeeze(1)

# criterion = nn.L1Loss()
# model = VisionTransformerDepth()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for images, depths in dataloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, depths)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)

#     epoch_loss = running_loss / len(dataloader.dataset)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# model.eval()
# total_loss = 0.0

# with torch.no_grad():
#     for images, depths in val_dataloader:
#         outputs = model(images)
#         loss = criterion(outputs, depths)
#         total_loss += loss.item() * images.size(0)

# val_loss = total_loss / len(val_dataloader.dataset)
# print(f"Validation Loss: {val_loss:.4f}")
