"""Fixed-checkpoint training-only generator ablation; validation at final step."""
import json, os, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import normalize_per_slice

mode=os.environ['LOSS_MODE']
if mode not in ('uniform','foreground10'): raise ValueError('unknown mode')
random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7);torch.set_num_threads(2)
device=torch.device('cuda:0')
ckpt=torch.load(os.environ['BASE_CHECKPOINT'],map_location='cpu',weights_only=True)
assert ckpt['model']['_extra_state']['current_task']==1 and not ckpt['model']['_extra_state']['task_open']
gen=SiteDiffusion(SiteDenoiser(6),sampling_steps=200).to(device)
gen.load_state_dict({k.removeprefix('generator.'):v for k,v in ckpt['model'].items() if k.startswith('generator.')})
opt=torch.optim.Adam(gen.parameters(),lr=0.0003)
opt.load_state_dict(ckpt['generator_optimizer'])
for state in opt.state.values():
    for key,val in state.items():
        if torch.is_tensor(val): state[key]=val.to(device)
for group in opt.param_groups: group['lr']=0.0003
loader=DataLoader(ProstateCL_Dataset('1','train',256),batch_size=8,shuffle=True)
it=iter(loader);gen.train()
for step in range(1000):
    try: x,y=next(it)
    except StopIteration: it=iter(loader);x,y=next(it)
    x=normalize_per_slice(x.to(device)); mask=y[:,None].float().to(device)
    t=torch.randint(1000,(len(x),),device=device);noise=torch.randn_like(x)
    a=gen.alpha_bars[t].view(-1,1,1,1);noisy=a.sqrt()*x+(1-a).sqrt()*noise
    pred=gen.denoiser(noisy,mask,t,torch.zeros(len(x),dtype=torch.long,device=device))
    weights=1+9*mask if mode=='foreground10' else torch.ones_like(mask)
    loss=((pred-noise).square()*weights).sum()/weights.sum()
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    if (step+1)%100==0: print('step',step+1,'loss',float(loss),flush=True)
# Fixed final model and fixed validation random stream, no checkpoint selection.
gen.eval();torch.manual_seed(701);torch.cuda.manual_seed_all(701)
items=[]
for x,y in DataLoader(ProstateCL_Dataset('1','val',256),batch_size=8,shuffle=False):
    for image,mask in zip(x,y):
        if bool((mask==1).any()):items.append((image,mask))
        if len(items)==24:break
    if len(items)==24:break
if len(items)!=24:raise RuntimeError('insufficient validation masks')
stats={str(t):{'fg_sq':0.,'bg_sq':0.,'fg':0,'bg':0} for t in (50,250,500,900)}
with torch.inference_mode():
    for off in range(0,24,8):
        x=normalize_per_slice(torch.stack([x for x,y in items[off:off+8]]).to(device))
        mask=torch.stack([y for x,y in items[off:off+8]]).to(device)[:,None].float()
        noise=torch.randn_like(x);ids=torch.zeros(len(x),dtype=torch.long,device=device);region=mask.bool()
        for t in (50,250,500,900):
            a=gen.alpha_bars[t];noisy=a.sqrt()*x+(1-a).sqrt()*noise
            pred=gen.denoiser(noisy,mask,torch.full((len(x),),t,dtype=torch.long,device=device),ids)
            if not torch.isfinite(pred).all():raise FloatingPointError('nonfinite validation')
            sq=(pred-noise).square();s=stats[str(t)]
            s['fg_sq']+=float(sq[region].sum());s['bg_sq']+=float(sq[~region].sum())
            s['fg']+=int(region.sum());s['bg']+=int((~region).sum())
results={t:{'foreground_noise_mse':s['fg_sq']/s['fg'],'background_noise_mse':s['bg_sq']/s['bg']} for t,s in stats.items()}
root=Path(os.environ['WORK_ROOT'])
torch.save({'generator':gen.state_dict(),'generator_optimizer':opt.state_dict(),'steps':1000,'mode':mode},root/'final.pth')
result={'mode':mode,'seed':7,'updates':1000,'train_split':'site1 train','validation_split':'first 24 nonempty site1 validation slices',
        'selection':'fixed final step','timesteps':results,'test_used':False,
        'caveat':'generator-only fixed checkpoint branch; forward noise MSE does not prove replay fidelity'}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
