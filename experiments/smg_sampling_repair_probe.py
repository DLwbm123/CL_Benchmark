"""Paired replay sampling qualification, fixed teacher; train masks only."""
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
torch.set_num_threads(2)
device = torch.device('cuda:0')
checkpoint = torch.load(os.environ['BASE_CHECKPOINT'], map_location='cpu', weights_only=True)
assert checkpoint['model']['_extra_state']['current_task'] == 1
assert not checkpoint['model']['_extra_state']['task_open']
model = resunet32('mid').to(device)
model.load_state_dict({k.removeprefix('net.core.'): v for k, v in checkpoint['model'].items() if k.startswith('net.core.')})
model.eval()
generator = SiteDiffusion(SiteDenoiser(6), sampling_steps=200).to(device)
generator.load_state_dict({k.removeprefix('generator.'): v for k, v in checkpoint['model'].items() if k.startswith('generator.')})
generator.eval()
masks=[]
for _, y in DataLoader(ProstateCL_Dataset('2','train',256),batch_size=8,shuffle=False):
    for item in y:
        if bool((item == 1).any()): masks.append(item)
        if len(masks) == 24: break
    if len(masks) == 24: break
if len(masks) != 24: raise RuntimeError('Insufficient nonempty masks')

def sample(mask, noise, clip_clean):
    x = noise.clone()
    ids = torch.zeros(len(mask),dtype=torch.long,device=device)
    times = torch.linspace(generator.steps-1,0,200,device=device).round().long().tolist()
    for i, step in enumerate(times):
        t = torch.full((len(mask),),step,dtype=torch.long,device=device)
        eps = generator.denoiser(x,mask,t,ids)
        alpha = generator.alpha_bars[step]
        clean = (x - (1-alpha).sqrt()*eps)/alpha.sqrt()
        if clip_clean: clean = clean.clamp(-1,1)
        if i+1 == len(times): x = clean
        else:
            next_alpha=generator.alpha_bars[times[i+1]]
            x=next_alpha.sqrt()*clean+(1-next_alpha).sqrt()*eps
    if not torch.isfinite(x).all(): raise FloatingPointError('nonfinite sample')
    return x.clamp(-1,1)

stats={m:{'tp':0,'pred':0,'target':0,'pixels':0,'saturated':0,'sensitivity':0.} for m in ('final_clamp','intermediate_clean_clamp')}
with torch.inference_mode():
    for start in range(0,24,8):
        mask=torch.stack(masks[start:start+8]).to(device)[:,None].float()
        noise=torch.randn_like(mask)
        for mode,clip in (('final_clamp',False),('intermediate_clean_clamp',True)):
            image=sample(mask,noise,clip)
            empty=sample(torch.zeros_like(mask),noise,clip)
            pred=model(image).argmax(1)
            target=mask[:,0].long()
            s=stats[mode]
            s['tp']+=int(((pred==1)&(target==1)).sum())
            s['pred']+=int((pred==1).sum()); s['target']+=int((target==1).sum())
            s['pixels']+=target.numel(); s['saturated']+=int((image.abs()>=.9999).sum())
            s['sensitivity']+=float((image-empty).abs().sum())
results={}
for mode,s in stats.items():
    results[mode]={'teacher_mask_foreground_micro_dice':2*s['tp']/(s['pred']+s['target']) if s['pred']+s['target'] else None,
                   'predicted_foreground_fraction':s['pred']/s['pixels'],'mask_foreground_fraction':s['target']/s['pixels'],
                   'saturation_fraction':s['saturated']/s['pixels'],'empty_mask_mean_abs_difference':s['sensitivity']/s['pixels']}
result={'seed':7,'samples':24,'sampling_steps':200,'split':'first 24 nonempty site2 train masks',
        'comparison':'same initial noise; clip estimated clean image at every DDIM step versus final clipping',
        'results':results,'test_used':False,'teacher_caveat':'Agreement is an imperfect proxy, not image fidelity.'}
Path(os.environ['WORK_ROOT']).joinpath('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
