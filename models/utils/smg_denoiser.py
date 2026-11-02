"""Paper-derived site-conditioned denoiser; architecture is an adaptation.

The SMG paper specifies 128 initial channels, 512-dimensional site/time
embeddings and attention at 16x16, but not all block/schedule details.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def time_embedding(t, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    phase = t.float()[:, None] * frequencies[None, :]
    return torch.cat((phase.sin(), phase.cos()), dim=1)


class Block(nn.Module):
    def __init__(self, inp, out, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out), out)
        self.embed = nn.Linear(embedding_dim, out)
        self.conv2 = nn.Conv2d(out, out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out), out)
        self.skip = nn.Conv2d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, embedding):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.embed(F.silu(embedding))[:, :, None, None]
        return self.skip(x) + self.norm2(self.conv2(F.silu(h)))


class SiteDenoiser(nn.Module):
    def __init__(self, sites, image_channels=1, mask_channels=1, width=128, embedding_dim=512):
        super().__init__()
        if sites < 1 or embedding_dim < 4 or embedding_dim % 2 or width % 32:
            raise ValueError('Invalid site count, width, or embedding dimension')
        self.embedding_dim = embedding_dim
        self.site_embeddings = nn.Embedding(sites, embedding_dim)
        self.time_mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.SiLU(),
                                      nn.Linear(embedding_dim, embedding_dim))
        self.input = nn.Conv2d(image_channels + mask_channels, width, 3, padding=1)
        self.down = nn.ModuleList([Block(width, width, embedding_dim) for _ in range(4)])
        self.mid = Block(width, width, embedding_dim)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.up = nn.ModuleList([Block(2 * width, width, embedding_dim) for _ in range(4)])
        self.output = nn.Conv2d(width, image_channels, 3, padding=1)

    def forward(self, noisy_image, mask, timestep, site_id):
        if noisy_image.shape[0] != mask.shape[0] or noisy_image.shape[-2:] != mask.shape[-2:]:
            raise ValueError('Image and conditioning mask shapes do not match')
        emb = self.time_mlp(time_embedding(timestep, self.embedding_dim)) + self.site_embeddings(site_id)
        x = self.input(torch.cat((noisy_image, mask.float()), dim=1))
        skips = []
        for block in self.down:
            x = block(x, emb)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.mid(x, emb)
        if x.shape[-2:] == (16, 16):
            b, c, h, w = x.shape
            seq = x.flatten(2).transpose(1, 2)
            x = x + self.attention(seq, seq, seq, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = block(torch.cat((x, skip), dim=1), emb)
        return self.output(F.silu(x))
