"""CNN CIT source optimizer; keep preallocated future query rows untouched."""
import torch
from models.utils.medical import MedicalModel


def configure_cit_optimizer(model):
    args=model.args
    if args.cit_weight_decay < 0 or args.cit_head_lr_mult <= 0:
        raise ValueError('Invalid CIT optimizer settings.')
    if args.cit_optimizer == 'SGD':
        # Explicit legacy mode for archived checkpoint diagnostics only.
        model.opt=torch.optim.SGD(model.net.parameters(),lr=args.lr)
    else:
        backbone=[];head=[]
        for name,param in model.net.named_parameters():
            (backbone if name.startswith('backbone.') else head).append(param)
        model.opt=torch.optim.AdamW([
            dict(params=backbone,lr_scale=1.),
            dict(params=head,lr=args.lr*args.cit_head_lr_mult,lr_scale=args.cit_head_lr_mult)],
            lr=args.lr,weight_decay=args.cit_weight_decay,betas=(.9,.999))
    model.scheduler=torch.optim.lr_scheduler.StepLR(model.opt,args.lr_decay_epoch,args.lr_decay_rate)


def cit_optimization_step(model,loss,**parts):
    query=model.net.queries.weight
    seen=int(model.net.seen_classes)
    future=query[seen:].detach().clone()
    value=MedicalModel.optimization_step(model,loss,**parts)
    with torch.no_grad():
        query[seen:].copy_(future)
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq','momentum_buffer'):
            state=model.opt.state.get(query,{})
            if key in state:state[key][seen:].zero_()
    return value
