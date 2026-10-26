"""S3R shape modules, re-expressed for the benchmark's 2D ResUNet.

The mask autoencoder follows the author's 16/32/64, 64-dimensional ACNN.
Adaptive pooling/interpolation remove its hard-coded 384x384 requirement.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.BatchNorm2d(cout),
                         nn.ReLU(), nn.Conv2d(cout, cout, 3, 1, 1),
                         nn.BatchNorm2d(cout), nn.ReLU())


class MaskAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(block(2, 16, 2), block(16, 32, 2),
                                     block(32, 64, 2), nn.Conv2d(64, 1, 3, 3, 1),
                                     nn.BatchNorm2d(1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((16, 16)), nn.Flatten(),
                                     nn.Linear(256, 64), nn.ReLU())
        self.decoder_fc = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 7, 3, 2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1))

    def encode(self, masks):
        return self.encoder(masks)

    def forward(self, masks):
        code = self.encode(masks)
        reconstruction = self.decoder(self.decoder_fc(code).reshape(-1, 1, 16, 16))
        reconstruction = F.interpolate(reconstruction, masks.shape[-2:],
                                       mode='bilinear', align_corners=False)
        return code, reconstruction


class ShapeResUnet(nn.Module):
    def __init__(self, backbone, dropout):
        super().__init__()
        if backbone.last.out_channels != 2:
            raise ValueError('S3R adapter currently supports binary Domain-CL only.')
        self.backbone = backbone
        # Retain the benchmark backbone; introduce the high-level dropout that
        # the original S3R backbone has, explicitly enabling MC uncertainty.
        for name in ('residual_conv_2', 'bridge', 'up_residual_conv1'):
            setattr(backbone, name, nn.Sequential(getattr(backbone, name),
                                                nn.Dropout2d(dropout)))
        self.distance_head = nn.Conv2d(backbone.last.in_channels, 1, 1)
        self.shape_encoder = MaskAutoencoder().requires_grad_(False)
        self.register_buffer('shape_ready', torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        self.shape_encoder.eval()
        return self

    def auxiliary(self, x):
        features = self.backbone.features(x)
        return self.backbone.logits(features), self.distance_head(features).tanh()

    def forward(self, x, task_index=None):
        return self.backbone(x)


def signed_distance(labels):
    """Normalized outside-minus-inside SDF with inner boundary zeroed.

    Empty masks return zero as upstream; full masks are handled without 0/0.
    Shapes and CPU/SciPy conversion are explicit, never retaining patient data.
    """
    result = np.zeros((labels.shape[0], 1, *labels.shape[-2:]), dtype=np.float32)
    for b, target in enumerate(labels.detach().cpu().numpy()):
        positive = target.astype(bool)
        if not positive.any():
            continue
        inside = distance_transform_edt(positive)
        outside = distance_transform_edt(~positive)
        def normalized(x):
            span = x.max() - x.min()
            return (x - x.min()) / span if span > 0 else np.zeros_like(x)
        sdf = normalized(outside) - normalized(inside)
        # Equivalent inner boundary using 4-neighbor connectivity.
        boundary = positive & ~binary_erosion(positive, border_value=1)
        sdf[boundary] = 0
        result[b, 0] = sdf
    return torch.from_numpy(result).to(device=labels.device, dtype=torch.float32)
