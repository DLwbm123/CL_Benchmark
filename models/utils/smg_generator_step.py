"""Train-only site-modulated diffusion update with synthetic old-site replay.

This component requires a pre-trained generator and is not a complete SMG
continual segmentation method or a substitute for its training lifecycle.
"""
import torch
from models.utils.smg_replay import generate_replay
from models.utils.smg_intensity import require_normalized


def generator_step(generator, optimizer, current_images, current_masks, current_site,
                   previous_sites=(), replay_batch=None):
    if (type(current_site) is not int or current_site < 0 or
            tuple(previous_sites) != tuple(range(current_site))):
        raise ValueError('Sites must form the preceding ordered stream')
    if (current_images.ndim != 4 or current_masks.ndim != 4 or
            current_images.shape[0] != current_masks.shape[0] or
            current_images.shape[-2:] != current_masks.shape[-2:]):
        raise ValueError('Current image/mask shapes do not match')
    if not torch.isfinite(current_images).all() or not torch.isfinite(current_masks).all():
        raise ValueError('Nonfinite input')
    require_normalized(current_images)
    ids = torch.full((current_images.shape[0],), current_site,
                     device=current_images.device, dtype=torch.long)
    images, masks, sites = current_images, current_masks, ids
    if previous_sites:
        if replay_batch is None:
            old_images, old_masks, old_ids = generate_replay(
                generator, current_masks, list(previous_sites), current_images.shape[1])
        else:
            old_images, old_labels, old_ids = replay_batch
            if (old_labels.ndim != 3 or old_labels.dtype != torch.long or
                    old_images.ndim != 4 or old_ids.ndim != 1 or old_ids.dtype != torch.long or
                    old_images.shape[0] == 0 or
                    old_images.shape[0] != old_labels.shape[0] or
                    old_images.shape[0] != old_ids.shape[0] or
                    old_images.shape[1:] != current_images.shape[1:] or
                    old_labels.shape[-2:] != current_masks.shape[-2:] or
                    ((old_labels != 0) & (old_labels != 1)).any() or
                    ((old_ids < 0) | (old_ids >= current_site)).any() or
                    not torch.isfinite(old_images).all()):
                raise ValueError('Invalid cached synthetic replay batch')
            old_masks = old_labels[:, None].float()
        images = torch.cat((current_images, old_images))
        masks = torch.cat((current_masks, old_masks))
        sites = torch.cat((ids, old_ids))
    generator.train()
    optimizer.zero_grad(set_to_none=True)
    loss = generator.loss(images, masks, sites)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError('Generator loss is nonfinite')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in generator.parameters() if p.grad is not None):
        raise FloatingPointError('Generator gradient is nonfinite')
    optimizer.step()
    return {'loss': float(loss.detach()), 'current_samples': current_images.shape[0],
            'synthetic_samples': images.shape[0] - current_images.shape[0]}
