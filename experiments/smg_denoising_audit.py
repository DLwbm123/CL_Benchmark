"""Fixed-noise forward-denoising audit, with mask ablation and validation only."""
import json, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import normalize_per_slice

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.set_num_threads(2)
device=torch.device('cuda:0')
ckpt=torch.load(os.environ['BASE_CHECKPOINT'],map_location='cpu',weights_only=True)
assert ckpt['model']['_extra_state']['current_task']==1
assert not ckpt['model']['_extra_state']['task_open']
gen=SiteDiffusion(SiteDenoiser(6),sampling_steps=200).to(device)
gen.load_state_dict({k.removeprefix('generator.'):v for k,v in ckpt['model'].items() if k.startswith('generator.')})
gen.eval()
items=[]
for x,y in DataLoader(ProstateCL_Dataset('1','val',256),batch_size=8,shuffle=False):
    for image,mask in zip(x,y):
        if bool((mask==1).any()): items.append((image,mask))
        if len(items)==24: break
    if len(items)==24: break
if len(items)!=24: raise RuntimeError('Insufficient nonempty validation masks')
stats={str(t):{'pixels':0,'fg_pixels':0,'noise_sq':0.,'correct_sq':0.,'empty_sq':0.,'difference_sq':0.,'fg_correct_sq':0.,'bg_correct_sq':0.} for t in (50,250,500,900)}
with torch.inference_mode():
    for off in range(0,24,8):
        x=normalize_per_slice(torch.stack([x for x,y in items[off:off+8]]).to(device))
        mask=torch.stack([y for x,y in items[off:off+8]]).to(device)[:,None].float()
        noise=torch.randn_like(x); sites=torch.zeros(len(x),dtype=torch.long,device=device)
        region=mask.bool()
        for t in (50,250,500,900):
            a=gen.alpha_bars[t]; noisy=a.sqrt()*x+(1-a).sqrt()*noise
            ts=torch.full((len(x),),t,dtype=torch.long,device=device)
            correct=gen.denoiser(noisy,mask,ts,sites)
            empty=gen.denoiser(noisy,torch.zeros_like(mask),ts,sites)
            if not torch.isfinite(correct).all() or not torch.isfinite(empty).all(): raise FloatingPointError('nonfinite output')
            sq=(correct-noise).square();s=stats[str(t)]
            s['pixels']+=x.numel();s['fg_pixels']+=int(region.sum());s['noise_sq']+=float(noise.square().sum())
            s['correct_sq']+=float(sq.sum());s['empty_sq']+=float((empty-noise).square().sum());s['difference_sq']+=float((correct-empty).square().sum())
            s['fg_correct_sq']+=float(sq[region].sum());s['bg_correct_sq']+=float(sq[~region].sum())
results={}
for t,s in stats.items():
    n=s['pixels']; f=s['fg_pixels']
    results[t]={'zero_prediction_noise_mse':s['noise_sq']/n,'mask_conditioned_noise_mse':s['correct_sq']/n,
                'empty_mask_noise_mse':s['empty_sq']/n,'mask_vs_empty_output_mse':s['difference_sq']/n,
                'foreground_noise_mse':s['fg_correct_sq']/f,'background_noise_mse':s['bg_correct_sq']/(n-f)}
result={'seed':7,'site':'1 validation','samples':24,'sample_rule':'first nonempty validation masks',
        'same_noise_for_timesteps_and_mask_controls':True,'timesteps':results,'test_used':False,
        'caveat':'Forward-noise prediction is a diagnostic and does not establish sample fidelity.'}
Path(os.environ['WORK_ROOT']).joinpath('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
