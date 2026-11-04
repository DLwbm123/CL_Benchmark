import os,json,torch
from pathlib import Path
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_denoiser import SiteDenoiser
from models.utils.smg_site_diffusion import SiteDiffusion
from models.utils.smg_intensity import normalize_per_slice
torch.manual_seed(701); torch.cuda.manual_seed_all(701); torch.set_num_threads(2); dev=torch.device('cuda:0')
c=torch.load(os.environ['BASE_CHECKPOINT'],map_location='cpu',weights_only=True)
assert c['model']['_extra_state']['current_task']==5 and not c['model']['_extra_state']['task_open']
net=resunet32('mid').to(dev).eval(); net.load_state_dict({k.removeprefix('net.core.'):v for k,v in c['model'].items() if k.startswith('net.core.')})
gen=SiteDiffusion(SiteDenoiser(6),sampling_steps=200).to(dev).eval(); gen.load_state_dict({k.removeprefix('generator.'):v for k,v in c['model'].items() if k.startswith('generator.')})
def add(s,x,y):
 z=net(x).argmax(1); s[0]+=int(((z==1)&(y==1)).sum()); s[1]+=int((z==1).sum()); s[2]+=int((y==1).sum()); s[3]+=y.numel()
def summary(s):
 a,b,c,d=s; return {'foreground_micro_dice':2*a/(b+c) if b+c else None,'predicted_foreground_fraction':b/d,'reference_foreground_fraction':c/d}
real={}; replay={}; masks=[]
with torch.inference_mode():
 for site in (1,2):
  stats=[0,0,0,0]
  for x,y in DataLoader(ProstateCL_Dataset(str(site),'val',256),batch_size=8): add(stats,normalize_per_slice(x.to(dev)),y.to(dev))
  real[str(site)]=summary(stats)
 for _,y in DataLoader(ProstateCL_Dataset('6','train',256),batch_size=8,shuffle=False):
  for m in y:
   if m.any(): masks.append(m)
   if len(masks)==24: break
  if len(masks)==24: break
 assert len(masks)==24
 labels=torch.stack(masks).to(dev)
 for site in (1,2):
  stats=[0,0,0,0]; sat=count=0
  for j in range(0,24,8):
   y=labels[j:j+8]; m=y[:,None].float(); ids=torch.full((len(y),),site-1,dtype=torch.long,device=dev); raw=gen.sample(m,ids,tuple(m.shape)); assert torch.isfinite(raw).all(); x=raw.clamp(-1,1); add(stats,x,y); sat+=int((x.abs()>=.9999).sum()); count+=x.numel()
  replay[str(site)]={**summary(stats),'saturation_fraction':sat/count}
res={'sample':'first 24 nonempty site6 train masks','seed':701,'real_validation_teacher':real,'synthetic_teacher_mask_agreement':replay,'qualification':'descriptive proxy only; no clinical fidelity or causal inference','test_used':False}
Path(os.environ['WORK_ROOT']).joinpath('result.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res),flush=True)
