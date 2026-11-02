"""CoNuSeg prototype helpers for an explicitly non-native ResUNet adaptation.

Not a complete continual method. Invalid/absent classes are tracked explicitly;
they must not be treated as measured zero prototypes.
"""
import torch
import torch.nn.functional as F


def batch_prototypes(features, labels, num_classes):
    if features.ndim != 4 or labels.ndim != 3 or features.shape[0] != labels.shape[0]:
        raise ValueError('Expected [B,C,H,W] features and [B,H,W] labels')
    if labels.shape[-2:] != features.shape[-2:]:
        labels = F.interpolate(labels[:, None].float(), size=features.shape[-2:], mode='nearest')[:, 0].long()
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError('Label outside output space')
    vectors = F.normalize(features, dim=1).permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    labels = labels.reshape(-1)
    sums = features.new_zeros(num_classes, features.shape[1]).index_add(0, labels, vectors)
    counts = torch.bincount(labels, minlength=num_classes)
    return sums / counts.clamp_min(1)[:, None], counts


def relation_distillation(current, teacher, valid, current_temp=0.2, past_temp=0.01):
    if current.shape != teacher.shape or valid.shape != (current.shape[0],):
        raise ValueError('Prototype shape mismatch')
    if min(current_temp, past_temp) <= 0:
        raise ValueError('Temperatures must be positive')
    current, teacher = current[valid], teacher[valid].detach()
    if current.shape[0] < 2:
        return current.sum() * 0
    diagonal = torch.eye(current.shape[0], device=current.device, dtype=torch.bool)
    logits = (current @ current.T / current_temp).masked_fill(diagonal, -torch.inf)
    old_logits = (teacher @ teacher.T / past_temp).masked_fill(diagonal, -torch.inf)
    mask = ~diagonal
    log_probs = F.log_softmax(logits, dim=1)[mask].view(current.shape[0], -1)
    targets = F.softmax(old_logits, dim=1)[mask].view(current.shape[0], -1)
    return -(targets * log_probs).sum(dim=1).mean()


def prototype_contrast(features, logits, labels, prototypes, counts, old_count,
                       temperature=0.1, hard=True):
    """Prototype-wise contrastive objective with the author's hard weighting.

    Global prototypes are detached running statistics; absent foreground
    prototypes and empty minibatches are excluded explicitly.
    """
    if temperature <= 0 or labels.ndim != 3:
        raise ValueError('Invalid contrast temperature or label dimensions')
    size = features.shape[-2:]
    if labels.shape[-2:] != size:
        labels = F.interpolate(labels[:, None].float(), size=size, mode='nearest')[:, 0].long()
    if logits.shape[-2:] != size:
        logits = F.interpolate(logits, size=size, mode='nearest')
    probabilities = logits.softmax(1)
    confidence, prediction = probabilities.max(1)
    vectors = F.normalize(features, dim=1)
    local = []
    classes = []
    for label in torch.unique(labels).tolist():
        if label <= 0 or label >= len(counts) or not counts[label]:
            continue
        positions = labels == label
        weight = torch.ones_like(confidence)
        if hard and label >= old_count:
            easy = positions & (prediction == label)
            weight = torch.where(easy, 1 - confidence, weight)
        selected = vectors.permute(0, 2, 3, 1)[positions] * weight[positions][:, None]
        if selected.numel():
            local.append(selected.mean(0))
            classes.append(label)
    available = torch.nonzero((counts > 0) & (torch.arange(len(counts), device=counts.device) > 0)).flatten()
    if not local or available.numel() < 2:
        return features.sum() * 0
    labels_out = torch.tensor(classes, device=features.device)
    targets = (available[None, :] == labels_out[:, None]).long().argmax(1)
    scores = torch.stack(local) @ prototypes.detach()[available].T / temperature
    return F.cross_entropy(scores, targets)
