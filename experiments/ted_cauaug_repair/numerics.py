"""Stable evaluation of the two-class softmax Dice+BCE objective."""
import torch
from torch.nn import functional as F

def stable_dice_bce(logits,labels):
    assert logits.shape[1]==2
    prob=logits.softmax(1);target=F.one_hot(labels,2).permute(0,3,1,2).float().to(logits.dtype)
    dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    # For exactly two softmax channels and one-hot targets, mean BCE equals CE.
    return dice+F.cross_entropy(logits,labels)

def stable_kd(logits,target,width_axis=False):
    return -(target*logits.log_softmax(1)).sum(-1 if width_axis else 1).mean()

def numeric_checks():
    torch.manual_seed(71)
    z=torch.randn(2,2,4,4,dtype=torch.float64,requires_grad=True);y=torch.randint(0,2,(2,4,4));p=z.softmax(1).clamp(1e-8,1-1e-8);t=F.one_hot(y,2).permute(0,3,1,2).double()
    old=1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)+F.binary_cross_entropy(p,t);new=stable_dice_bce(z,y)
    go=torch.autograd.grad(old,z,retain_graph=True)[0];gn=torch.autograd.grad(new,z)[0]
    assert torch.allclose(old,new,atol=1e-12,rtol=1e-12) and torch.allclose(go,gn,atol=1e-12,rtol=1e-12)
    target=torch.randn_like(z).softmax(1)
    for width in [False,True]:
        a=-(target*z.softmax(1).log()).sum(-1 if width else 1).mean();b=stable_kd(z,target,width);assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    z=torch.tensor([[[[100.]], [[-100.]]]],requires_grad=True);y=torch.ones(1,1,1,dtype=torch.long);p=z.softmax(1).clamp(1e-8,1-1e-8);t=F.one_hot(y,2).permute(0,3,1,2).float()
    old=1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)+F.binary_cross_entropy(p,t);new=stable_dice_bce(z,y)
    go=torch.autograd.grad(old,z,retain_graph=True)[0];gn=torch.autograd.grad(new,z)[0]
    assert (go==0).all() and gn.abs().sum()>0 and torch.isfinite(gn).all()
    return {'moderate_value_gradient_equivalence':True,'kd_reduction_axes_preserved':True,'extreme_wrong_prediction_old_zero_gradient':True,'stable_nonzero_finite_gradient':True}
