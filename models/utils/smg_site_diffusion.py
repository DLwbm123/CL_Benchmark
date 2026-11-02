"""Mask-conditioned, site-modulated diffusion primitives for SMG adaptation.

The denoiser is supplied by the caller and must accept (noisy_image, mask,
timestep, site_id). This module does not provide the paper's U-Net, a replay
buffer, or the sequential training lifecycle; it is not a complete method.
"""
import torch
from torch import nn


class SiteDiffusion(nn.Module):
    def __init__(self, denoiser, steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        if steps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError('Invalid diffusion schedule')
        self.denoiser = denoiser
        self.steps = steps
        betas = torch.linspace(beta_start, beta_end, steps)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', torch.cumprod(1 - betas, dim=0))

    def loss(self, image, mask, site_id, timestep=None, noise=None):
        if image.shape[0] != mask.shape[0] or image.shape[0] != site_id.shape[0]:
            raise ValueError('Batch sizes must match')
        batch = image.shape[0]
        if timestep is None:
            timestep = torch.randint(self.steps, (batch,), device=image.device)
        if noise is None:
            noise = torch.randn_like(image)
        if noise.shape != image.shape or timestep.shape != (batch,):
            raise ValueError('Invalid noise or timestep shape')
        scale = self.alpha_bars[timestep].view(batch, *([1] * (image.ndim - 1)))
        noisy = scale.sqrt() * image + (1 - scale).sqrt() * noise
        predicted = self.denoiser(noisy, mask, timestep, site_id)
        if predicted.shape != image.shape:
            raise ValueError('Denoiser output shape must match image')
        return (predicted - noise).square().mean()

    @torch.no_grad()
    def sample(self, mask, site_id, image_shape, noise=None):
        if image_shape[0] != mask.shape[0] or image_shape[0] != site_id.shape[0]:
            raise ValueError('Batch sizes must match')
        x = torch.randn(image_shape, device=mask.device) if noise is None else noise.clone()
        if tuple(x.shape) != tuple(image_shape):
            raise ValueError('Invalid initial noise shape')
        for step in reversed(range(self.steps)):
            t = torch.full((image_shape[0],), step, device=x.device, dtype=torch.long)
            prediction = self.denoiser(x, mask, t, site_id)
            if prediction.shape != x.shape:
                raise ValueError('Denoiser output shape must match image')
            beta = self.betas[step]
            alpha = 1 - beta
            x = (x - beta / (1 - self.alpha_bars[step]).sqrt() * prediction) / alpha.sqrt()
            if step:
                posterior_variance = beta * (1 - self.alpha_bars[step - 1]) / (1 - self.alpha_bars[step])
                x = x + posterior_variance.sqrt() * torch.randn_like(x)
        return x
