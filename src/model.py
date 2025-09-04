import torch
from torch import nn, Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# Helper modules
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# 2D ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (class token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

# 3D Convolutional Decoder
class Decoder3D(nn.Module):
    def __init__(self, *, input_dim, output_size=256, initial_size=4, latent_channels=512):
        super().__init__()
        self.initial_shape = (latent_channels, initial_size, initial_size, initial_size)
        self.linear = nn.Linear(input_dim, int(torch.prod(torch.tensor(self.initial_shape))))

        layers = []

        # Calculate the number of upsampling layers needed
        num_upsamples = int(np.log2(output_size // initial_size))

        in_channels = latent_channels

        # Dynamically create upsampling layers
        for i in range(num_upsamples):
            out_channels = in_channels // 2 if in_channels > 16 else 16 # Don't go below 16 channels until the end
            layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels

        # Final layer to get to 1 channel output
        layers.append(nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid()) # Use Sigmoid for output normalized to [0, 1]

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), *self.initial_shape)
        x = self.upsample(x)
        return x

# Main Model: XrayTo3D
class XrayTo3D(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 volume_size,
                 enc_dim=512,
                 enc_depth=6,
                 enc_heads=8,
                 enc_mlp_dim=1024):
        super().__init__()

        self.frontal_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=enc_dim,
            depth=enc_depth,
            heads=enc_heads,
            mlp_dim=enc_mlp_dim,
            channels=1
        )
        self.lateral_encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=enc_dim,
            depth=enc_depth,
            heads=enc_heads,
            mlp_dim=enc_mlp_dim,
            channels=1
        )

        self.decoder = Decoder3D(
            input_dim=enc_dim * 2, # Concatenated features from two encoders
            output_size=volume_size
        )

    def forward(self, frontal_img, lateral_img):
        frontal_features = self.frontal_encoder(frontal_img)
        lateral_features = self.lateral_encoder(lateral_img)

        # Fusion by concatenation
        fused_features = torch.cat([frontal_features, lateral_features], dim=1)

        # Decode to 3D volume
        recon_volume = self.decoder(fused_features)
        return recon_volume

if __name__ == '__main__':
    # Example Usage and Verification

    # --- Test with 32x32 configuration ---
    print("--- Testing 32x32 configuration ---")
    model_32 = XrayTo3D(
        image_size=32,
        patch_size=4,
        volume_size=32,
        enc_dim=512,
        enc_depth=3,
        enc_heads=4,
        enc_mlp_dim=512,
    )

    frontal_img_32 = torch.randn(2, 1, 32, 32)
    lateral_img_32 = torch.randn(2, 1, 32, 32)
    output_volume_32 = model_32(frontal_img_32, lateral_img_32)

    print(f"Input shape (frontal): {frontal_img_32.shape}")
    print(f"Output volume shape: {output_volume_32.shape}")
    expected_shape_32 = (2, 1, 32, 32, 32)
    assert output_volume_32.shape == expected_shape_32, f"Shape mismatch! Expected {expected_shape_32}, got {output_volume_32.shape}"
    print("32x32 configuration test PASSED.")

    # --- Test with 256x256 configuration ---
    print("\n--- Testing 256x256 configuration ---")
    model_256 = XrayTo3D(
        image_size=256,
        patch_size=16,
        volume_size=256,
        enc_dim=1024,
        enc_depth=6,
        enc_heads=8,
        enc_mlp_dim=2048,
    )

    frontal_img_256 = torch.randn(1, 1, 256, 256)
    lateral_img_256 = torch.randn(1, 1, 256, 256)
    output_volume_256 = model_256(frontal_img_256, lateral_img_256)

    print(f"Input shape (frontal): {frontal_img_256.shape}")
    print(f"Output volume shape: {output_volume_256.shape}")
    expected_shape_256 = (1, 1, 256, 256, 256)
    assert output_volume_256.shape == expected_shape_256, f"Shape mismatch! Expected {expected_shape_256}, got {output_volume_256.shape}"
    print("256x256 configuration test PASSED.")
