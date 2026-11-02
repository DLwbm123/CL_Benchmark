"""One paper-derived dual-meta step using current-mask-conditioned replay.

This is not a complete SMG lifecycle: generator pretraining, sequential
updates and persistent full checkpoints must be handled by the caller.
"""
import torch
from models.utils.smg_dual_meta import first_order_dual_meta
from models.utils.smg_replay import generate_replay
from models.utils.smg_intensity import require_normalized


def segmentation_step(model, optimizer, loss_fn, generator, images, labels,
                      current_site, gamma=5e-5, beta=5e-4, replay_batch=None):
    if type(current_site) is not int or current_site < 1:
        raise ValueError('Dual-meta replay requires an incremental site')
    if (images.ndim != 4 or labels.ndim != 3 or images.shape[0] != labels.shape[0]
            or images.shape[-2:] != labels.shape[-2:]):
        raise ValueError('Invalid current image/label shapes')
    if labels.dtype != torch.long or ((labels != 0) & (labels != 1)).any():
        raise ValueError('Expected binary integer labels')
    if not torch.isfinite(images).all():
        raise ValueError('Inputs must be finite')
    require_normalized(images)
    masks = labels[:, None].float()
    if replay_batch is None:
        replay_images, replay_masks, _ = generate_replay(generator, masks, list(range(current_site)),
                                                         image_channels=images.shape[1])
        replay_labels = replay_masks[:, 0].long()
    else:
        replay_images, replay_labels, replay_sites = replay_batch
        if (replay_images.ndim != 4 or replay_labels.ndim != 3 or replay_sites.ndim != 1 or
                replay_sites.dtype != torch.long or replay_images.shape[0] == 0 or
                replay_labels.dtype != torch.long or replay_images.shape[0] != replay_labels.shape[0] or
                replay_images.shape[0] != replay_sites.shape[0] or replay_images.shape[1:] != images.shape[1:] or
                replay_labels.shape[-2:] != labels.shape[-2:] or
                ((replay_labels != 0) & (replay_labels != 1)).any() or
                ((replay_sites < 0) | (replay_sites >= current_site)).any() or
                not torch.isfinite(replay_images).all()):
            raise ValueError('Invalid cached synthetic replay batch')
    combined_images = torch.cat((images, replay_images))
    combined_labels = torch.cat((labels, replay_labels))
    order = torch.randperm(combined_images.shape[0], device=images.device)
    midpoint = combined_images.shape[0] // 2
    if midpoint == 0 or midpoint == combined_images.shape[0]:
        raise ValueError('Need nonempty virtual train and test splits')
    model.train()
    # The functional inner loops clone buffers. Update live BN statistics once
    # on the current training batch; generated data never update live buffers.
    with torch.no_grad():
        if not torch.isfinite(model(images)).all():
            raise FloatingPointError('Current logits are nonfinite')
    total, grads = first_order_dual_meta(
        model, loss_fn, (images, labels), (replay_images, replay_labels),
        (combined_images[order[:midpoint]], combined_labels[order[:midpoint]]),
        (combined_images[order[midpoint:]], combined_labels[order[midpoint:]]),
        gamma=gamma, beta=beta)
    optimizer.zero_grad(set_to_none=True)
    for name, param in model.named_parameters():
        param.grad = grads[name]
    optimizer.step()
    return {'loss': float(total), 'current_samples': images.shape[0],
            'synthetic_samples': replay_images.shape[0]}
