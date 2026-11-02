"""Train-only pruning acceptance helper for a non-native clNet adaptation.

The author implementation evaluates pruning on validation data. This helper
requires an explicitly train-derived score instead, so its behavior cannot be
reported as an author-native reproduction. The caller owns score provenance.
"""
import copy
import math


def prune_with_train_gate(decoder, masks, optimizer, target, train_score,
                          baseline, max_drop=0.02, max_attempts=8):
    if not callable(train_score) or not math.isfinite(float(baseline)):
        raise ValueError('A finite training-only baseline and score function are required')
    if not 0 <= max_drop < 1 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('Invalid pruning acceptance configuration')
    old_state = copy.deepcopy(decoder.state_dict())
    old_masks = copy.deepcopy(masks.state_dict())
    total = sum(getattr(masks, 'mask_' + str(i)).numel() for i in range(len(masks.names)))
    old_pruned = sum(int((~getattr(masks, 'mask_' + str(i))).sum()) for i in range(len(masks.names)))
    current = old_pruned / total
    if not current <= target < 1:
        raise ValueError('Target must not reduce existing sparsity')
    candidate = target
    try:
        for attempt in range(1, max_attempts + 1):
            decoder.load_state_dict(old_state)
            masks.load_state_dict(old_masks)
            masks.set_sparsity(decoder, candidate)
            score = float(train_score())
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite training pruning score')
            if score >= baseline - max_drop:
                masks.apply(decoder, optimizer)
                return {'accepted': True, 'requested': target,
                        'accepted_sparsity': candidate, 'attempts': attempt,
                        'train_baseline': baseline, 'train_score': score}
            candidate = (current + candidate) / 2
    except Exception:
        decoder.load_state_dict(old_state)
        masks.load_state_dict(old_masks)
        raise
    decoder.load_state_dict(old_state)
    masks.load_state_dict(old_masks)
    return {'accepted': False, 'requested': target, 'accepted_sparsity': current,
            'attempts': max_attempts, 'train_baseline': baseline}
