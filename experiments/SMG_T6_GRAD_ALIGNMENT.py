import torch
from torch.func import functional_call

def alignment(model, loss_fn, current, replay):
    params=dict(model.named_parameters())
    if not params: raise ValueError('empty model')
    buffers=dict(model.named_buffers())
    def calc(batch):
        x,y=batch
        b={k:v.detach().clone() for k,v in buffers.items()}
        value=loss_fn(functional_call(model,(params,b),(x,)),y)
        if value.ndim or not torch.isfinite(value): raise FloatingPointError('nonfinite loss')
        g=torch.autograd.grad(value,tuple(params.values()))
        if not all(torch.isfinite(v).all() for v in g): raise FloatingPointError('nonfinite gradient')
        return value.detach(), torch.cat([v.detach().flatten() for v in g]).double()
    cl,cg=calc(current);rl,rg=calc(replay)
    cn=cg.norm();rn=rg.norm()
    if not torch.isfinite(cn) or not torch.isfinite(rn) or min(float(cn),float(rn))<=0:
        raise FloatingPointError('invalid norm')
    cosine=torch.dot(cg,rg)/(cn*rn)
    if not torch.isfinite(cosine) or abs(float(cosine))>1.0000001:raise FloatingPointError('invalid cosine')
    return {'current_loss':float(cl),'replay_loss':float(rl),'current_grad_norm':float(cn),'replay_grad_norm':float(rn),'cosine':float(cosine)}

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
    if not os.environ.get('CL_DATA_ROOT'):
        raise RuntimeError('CL_DATA_ROOT must be explicitly supplied')
    if not (Path(os.environ['CL_DATA_ROOT']) / 'Domain_Prostate' / 'I2CVB.h5').is_file():
        raise FileNotFoundError('Required training dataset is not accessible')
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
        x=normalize_per_slice(x.to(dev)); y=y.to(dev)
        idx=torch.randint(len(buffer.images),(len(x),),generator=replay_rng)
        rx=buffer.images[idx].to(dev); ry=buffer.labels[idx].to(dev)
        with torch.no_grad():model(x)
        row=alignment(model,objective,(x,y),(rx,ry))
        row['prompt_counts']=[int((buffer.sites[idx]==s).sum()) for s in range(5)]
        rows.append(row)
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(baseline[k])
        if not all(torch.equal(v,baseline[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('model state was modified')
    if len(rows)!=3:raise RuntimeError('unexpected batch count')
    out={'batches':rows,'split':'site6 train first 24','training':False,'test_used':False,'validation_used':False,'status':'complete'}
    Path(os.environ['WORK_ROOT'],'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('audit completed',flush=True)
