import torch
from torch.func import functional_call

def gradient(model, loss_fn, inner_batch, outer_batch, rate, clip=1., exact=False):
    p = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    if not p or rate <= 0 or clip <= 0: raise ValueError('invalid arguments')
    def loss(params, batch):
        values = {k: v.detach().clone() for k, v in buffers.items()}
        result = loss_fn(functional_call(model, (params, values), (batch[0],)), batch[1])
        if result.ndim or not torch.isfinite(result): raise FloatingPointError('nonfinite loss')
        return result
    inner = loss(p, inner_batch)
    g = torch.autograd.grad(inner, tuple(p.values()), create_graph=exact, retain_graph=True)
    norm = torch.linalg.vector_norm(torch.stack([v.norm() for v in g]))
    if not torch.isfinite(norm) or norm > 1e6: raise FloatingPointError('invalid inner norm')
    scale = clip / norm.clamp_min(clip)
    if not exact:
        g = tuple(v.detach() for v in g)
        scale = scale.detach()
    adapted = {k: v - rate * scale * dv for (k, v), dv in zip(p.items(), g)}
    total = inner + loss(adapted, outer_batch)
    grad = torch.autograd.grad(total, tuple(p.values()))
    if not all(torch.isfinite(v).all() for v in grad): raise FloatingPointError('nonfinite gradient')
    return total.detach(), tuple(v.detach() for v in grad), norm.detach(), scale.detach()

def dual_gradient(model, loss_fn, current, replay, meta_train, meta_test, exact=False):
    first = gradient(model, loss_fn, current, replay, 5e-5, 1., exact)
    second = gradient(model, loss_fn, meta_train, meta_test, 5e-4, 1., exact)
    combined = tuple(a+b for a,b in zip(first[1],second[1]))
    if not all(torch.isfinite(t).all() for t in combined): raise FloatingPointError('nonfinite combined gradient')
    return first[0]+second[0], combined, (first[2],second[2]), (first[3],second[3])

if __name__ == "__main__":
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
    torch.cuda.set_per_process_memory_fraction(.24)
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
    baseline={k:v.detach().clone() for k,v in model.state_dict().items()}
    for x,y in samples:
        x=normalize_per_slice(x.to(dev));y=y.to(dev)
        idx=torch.randint(len(buffer.images),(len(x),),generator=replay_rng)
        rx=buffer.images[idx].to(dev);ry=buffer.labels[idx].to(dev)
        xx=torch.cat((x,rx));yy=torch.cat((y,ry));order=torch.randperm(len(xx),generator=split_rng).to(dev);mid=len(order)//2
        with torch.no_grad(): model(x)
        args=((x,y),(rx,ry),(xx[order[:mid]],yy[order[:mid]]),(xx[order[mid:]],yy[order[mid:]]))
        fo=dual_gradient(model,objective,*args,False)
        exact=dual_gradient(model,objective,*args,True)
        f=torch.cat([z.flatten() for z in fo[1]]).double();e=torch.cat([z.flatten() for z in exact[1]]).double()
        fn=f.norm();en=e.norm()
        if min(float(fn),float(en))<=0: raise FloatingPointError('zero gradient')
        rows.append({'loss_first_order':float(fo[0]),'loss_second_order':float(exact[0]),'first_order_norm':float(fn),'second_order_norm':float(en),'cosine':float(torch.dot(f,e)/(fn*en)),'relative_difference':float((f-e).norm()/fn),'inner_norms':[float(v) for v in fo[2]],'clip_scales':[float(v) for v in fo[3]]})
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(baseline[k])
        assert all(torch.equal(v,baseline[k]) for k,v in model.state_dict().items())
    out={'batches':rows,'split':'site6 train first 24','training':False,'test_used':False,'validation_used':False,'status':'complete'}
    Path(os.environ['WORK_ROOT'],'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('audit completed',flush=True)
