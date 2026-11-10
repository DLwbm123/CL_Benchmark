"""Read-only gradient measurements on preregistered training steps."""
import math
import torch

def measure(net,supervised,distill,weight):
    params=[p for p in net.parameters() if p.requires_grad]
    gs=torch.autograd.grad(supervised,params,retain_graph=True,allow_unused=True)
    gd=torch.autograd.grad(distill,params,retain_graph=True,allow_unused=True)
    s2=d2=dot=0.
    for a,b in zip(gs,gd):
        if a is not None:s2+=float(a.detach().double().square().sum())
        if b is not None:d2+=float(b.detach().double().square().sum())
        if a is not None and b is not None:dot+=float((a.detach().double()*b.detach().double()).sum())
    sn=math.sqrt(s2);dn=math.sqrt(d2)
    out={'supervised_norm':sn,'distillation_norm':dn,'weighted_distillation_norm':weight*dn,
         'weighted_ratio':weight*dn/sn if sn>0 else None,'cosine':dot/(sn*dn) if sn*dn>0 else None,
         'supervised_loss':float(supervised.detach()),'distillation_loss':float(distill.detach()),'weight':weight}
    assert all(v is None or math.isfinite(v) for v in out.values())
    return out
