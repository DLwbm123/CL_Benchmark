"""Synthetic replay pairs using current-domain masks and prior site prompts.

This is a component, not the full SMG training lifecycle. No old raw image or
old mask is read; generated tensors must remain within the private runtime.
"""
import torch


@torch.no_grad()
def generate_replay(generator, current_masks, previous_sites, image_channels=1):
    if current_masks.ndim != 4 or not current_masks.is_floating_point():
        raise ValueError('Current masks must be float [B,1,H,W]')
    if current_masks.shape[1] != 1 or not current_masks.shape[0]:
        raise ValueError('Expected a nonempty single-channel mask batch')
    if not torch.isfinite(current_masks).all():
        raise ValueError('Current masks must be finite')
    if not previous_sites or any(type(site) is not int or site < 0 for site in previous_sites):
        raise ValueError('Provide previous nonnegative site indices')
    if len(set(previous_sites)) != len(previous_sites) or image_channels < 1:
        raise ValueError('Sites must be distinct and image channels positive')
    training = generator.training
    generator.eval()
    images, labels, sites = [], [], []
    try:
        for site in previous_sites:
            ids = torch.full((current_masks.shape[0],), site, device=current_masks.device, dtype=torch.long)
            shape = (current_masks.shape[0], image_channels, *current_masks.shape[2:])
            image = generator.sample(current_masks, ids, shape)
            if tuple(image.shape) != shape or not torch.isfinite(image).all():
                raise FloatingPointError('Invalid generated replay image')
            # Match the fixed training intensity range before replay is reused.
            image = image.clamp(-1, 1)
            images.append(image)
            labels.append(current_masks.clone())
            sites.append(ids)
    finally:
        generator.train(training)
    return torch.cat(images), torch.cat(labels), torch.cat(sites)


class SMGReplayBuffer:
    """Private CPU buffer built from current masks and previous site prompts.

    Generation occurs before incremental training; callers choose a frozen,
    train-only mask stream and a hard byte budget. No data are published.
    """
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('Positive byte budget required')
        self.max_bytes = max_bytes
        self.images = self.labels = self.sites = None

    @torch.no_grad()
    def build(self, generator, masks, previous_sites, image_channels=1, batch_size=1):
        if self.images is not None:
            raise RuntimeError('Create a new buffer per task')
        if masks.ndim != 4 or masks.shape[1] != 1 or masks.dtype != torch.float32:
            raise ValueError('Expected train masks [B,1,H,W] as float32')
        if not torch.isfinite(masks).all() or ((masks != 0) & (masks != 1)).any():
            raise ValueError('Train masks must be binary and finite')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Positive generation batch size required')
        if not previous_sites or len(set(previous_sites)) != len(previous_sites):
            raise ValueError('Previous sites must be nonempty and distinct')
        count = masks.shape[0] * len(previous_sites)
        h, w = masks.shape[-2:]
        # Float32 images, int64 labels and int64 site IDs; reserve 2x for
        # temporary lists/concatenation and fail before generating anything.
        required = count * (image_channels * h * w * 4 + h * w * 8 + 8)
        if count < 1 or 2 * required > self.max_bytes:
            raise MemoryError('Replay buffer would exceed its reserved byte budget')
        images, labels, sites = [], [], []
        try:
            device = next(generator.parameters()).device
        except StopIteration:
            device = masks.device
        for start in range(0, masks.shape[0], batch_size):
            chunk = masks[start:start + batch_size].to(device)
            x, y, s = generate_replay(generator, chunk, previous_sites, image_channels)
            images.append(x.to(device='cpu', dtype=torch.float32))
            labels.append(y[:, 0].to(device='cpu', dtype=torch.long))
            sites.append(s.cpu())
        self.images, self.labels, self.sites = torch.cat(images), torch.cat(labels), torch.cat(sites)

    def sample(self, batch_size, device):
        if self.images is None or type(batch_size) is not int or batch_size < 1:
            raise ValueError('Build a buffer and request a positive batch size')
        indices = torch.randint(len(self.images), (batch_size,))
        return (self.images[indices].to(device), self.labels[indices].to(device),
                self.sites[indices].to(device))

    def state_dict(self):
        if self.images is None:
            raise RuntimeError('Cannot checkpoint an empty replay buffer')
        return {'images': self.images, 'labels': self.labels, 'sites': self.sites}

    def load_state_dict(self, state):
        x, y, s = state['images'], state['labels'], state['sites']
        if (not all(torch.is_tensor(t) and t.device.type == 'cpu' for t in (x, y, s)) or
                x.dtype != torch.float32 or y.dtype != torch.long or s.dtype != torch.long or
                x.ndim != 4 or y.ndim != 3 or s.ndim != 1 or x.shape[0] == 0 or
                x.shape[0] != y.shape[0] or x.shape[0] != s.shape[0] or
                x.shape[-2:] != y.shape[-2:] or not torch.isfinite(x).all() or
                ((y != 0) & (y != 1)).any() or (s < 0).any()):
            raise ValueError('Invalid replay checkpoint tensors')
        size = sum(t.numel() * t.element_size() for t in (x, y, s))
        if size * 2 > self.max_bytes:
            raise MemoryError('Replay checkpoint exceeds its reserved byte budget')
        self.images, self.labels, self.sites = x, y, s
