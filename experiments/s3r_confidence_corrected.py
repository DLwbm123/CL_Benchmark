"""Faithful sign of the source S3R confidence, retaining zero-span safeguard."""
import torch

def confidence_mask(mc_probabilities):
 average=mc_probabilities.mean(0)
 confidence=torch.exp((average*(average+1e-6).log()).sum(1,keepdim=True))
 low=confidence.amin(dim=(-2,-1),keepdim=True)
 span=confidence.amax(dim=(-2,-1),keepdim=True)-low
 return torch.where(span>0,(confidence-low)/span.clamp_min(1e-12),torch.ones_like(confidence))
