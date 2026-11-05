"""Read-only comparison of clipped and unclipped first-order inner steps."""
import torch
from torch.func import functional_call

def measure(model, loss_fn, inner_batch, outer_batch, rate):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    def evaluate(p, batch):
        x, y = batch
        cloned = {k: v.detach().clone() for k,v in buffers.items()}
        value = loss_fn(functional_call(model, (p, cloned), (x,)), y)
        if value.ndim != 0 or not torch.isfinite(value):
            raise FloatingPointError('nonfinite loss')
        return value
    loss = evaluate(params, inner_batch)
    grads = torch.autograd.grad(loss, tuple(params.values()))
    norm = torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads]))
    if not torch.isfinite(norm): raise FloatingPointError('nonfinite gradient')
    scale = norm.clamp_min(1).reciprocal()
    results = {'inner_loss':float(loss.detach()),'gradient_norm':float(norm), 'clip_scale':float(scale)}
    with torch.no_grad():
        results['outer_initial']=float(evaluate(params,outer_batch))
        pnorm=torch.linalg.vector_norm(torch.stack([p.detach().norm() for p in params.values()]))
        for label,mult in [('clip1',scale),('unclipped',torch.ones_like(scale))]:
            changed={k:p-rate*mult*g.detach() for (k,p),g in zip(params.items(),grads)}
            if not all(torch.isfinite(v).all() for v in changed.values()): raise FloatingPointError('nonfinite adapted parameters')
            results['outer_'+label]=float(evaluate(changed,outer_batch))
            results['relative_step_'+label]=float(rate*mult*norm/pnorm.clamp_min(1e-30))
    return results

if __name__ == '__main__':
    import os,json,random
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader,Subset
    from torch.nn import functional as F
    from backbone.ResUnet import resunet32
    from datasets.seq_prostate import ProstateCL_Dataset
    from models.utils.smg_denoiser import SiteDenoiser
    from models.utils.smg_site_diffusion import SiteDiffusion
    from models.utils.smg_intensity import normalize_per_slice
    from models.utils.smg_replay import SMGReplayBuffer
    random.seed(701); np.random.seed(701); torch.manual_seed(701); torch.cuda.manual_seed_all(701); torch.set_num_threads(2)
    dev=torch.device('cuda:0')
    ck=torch.load(os.environ['BASE_CHECKPOINT'],map_location='cpu',weights_only=True)
    assert ck['model']['_extra_state']['current_task']==5 and not ck['model']['_extra_state']['task_open']
    model=resunet32('mid').to(dev)
    model.load_state_dict({k.removeprefix('net.core.'):v for k,v in ck['model'].items() if k.startswith('net.core.')})
    gen=SiteDiffusion(SiteDenoiser(6),sampling_steps=200).to(dev)
    gen.load_state_dict({k.removeprefix('generator.'):v for k,v in ck['model'].items() if k.startswith('generator.')})
    ds=Subset(ProstateCL_Dataset('6','train',256),list(range(24)))
    samples=list(DataLoader(ds,batch_size=8,shuffle=False))
    masks=torch.cat([y[:,None].float() for _,y in samples])
    buffer=SMGReplayBuffer(1024**3);buffer.build(gen,masks,list(range(5)),batch_size=8)
    replay_rng=torch.Generator().manual_seed(703);split_rng=torch.Generator().manual_seed(702)
    def objective(logits,y):
        fg=logits.softmax(dim=1)[:,1]; target=(y==1).float()
        return F.cross_entropy(logits,y)+1-(2*(fg*target).sum()+1)/(fg.sum()+target.sum()+1)
    rows=[];model.train()
    baseline={k:v.detach().clone() for k,v in model.named_buffers()}
    for x,y in samples:
        x=normalize_per_slice(x.to(dev)); y=y.to(dev)
        idx=torch.randint(len(buffer.images),(len(x),),generator=replay_rng)
        rx=buffer.images[idx].to(dev);ry=buffer.labels[idx].to(dev)
        xx=torch.cat((x,rx));yy=torch.cat((y,ry)); order=torch.randperm(len(xx),generator=split_rng).to(dev); mid=len(order)//2
        with torch.no_grad(): model(x)
        a=measure(model,objective,(x,y),(rx,ry),5e-5)
        b=measure(model,objective,(xx[order[:mid]],yy[order[:mid]]),(xx[order[mid:]],yy[order[mid:]]),5e-4)
        rows.append({'orientational':a,'arbitrary':b})
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(baseline[k])
    out={'batches':rows,'split':'site6 train first 24','training':False,'test_used':False,'validation_used':False,'status':'complete'}
    Path(os.environ['WORK_ROOT'],'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('audit completed',flush=True)
