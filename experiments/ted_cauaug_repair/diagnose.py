import os,json,copy,random,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from runner import make_model as make_cau,dataset as cau_dataset,finite,atomic_json
from ted_runner import make_model as make_ted
from shared import dataset as ted_dataset
from numerics import stable_dice_bce,numeric_checks
r=Path(os.environ['WORK_ROOT']);method=os.environ['ARM_METHOD'];seed=int(os.environ['ARM_SEED']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.18)
checks=numeric_checks();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
if method=='ted':net,_,_,_=make_ted();agent=None;get_data=ted_dataset
else:net,_,agent=make_cau('cauaug',os.environ['REFERENCE_ROOT']);get_data=cau_dataset
cp=torch.load(Path(os.environ['OLD_ROOT'])/f'{method}_s{seed}_latest.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['model']);net.train();saved={k:v.detach().clone() for k,v in net.state_dict().items()}
count=wrong=zero_old=zero_new=0;old_grad=new_grad=0.;max_logit=0.
for site in range(1,7):
 ds=get_data(os.environ['DATA_ROOT'],site,'train',method=method)
 indices=np.linspace(0,len(ds)-1,min(16,len(ds)),dtype=int)
 for index in indices:
  x,y=ds[int(index)];x=x[None].cuda();y=y[None].cuda()
  with torch.no_grad():
   if agent is not None:x=agent.cross_kernel_network(x)
   z=net(x).detach()
  finite(z,'logits');max_logit=max(max_logit,float(z.abs().max()));z.requires_grad_(True);p=z.softmax(1).clamp(1e-8,1-1e-8);t=F.one_hot(y,2).permute(0,3,1,2).float()
  old=1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)+F.binary_cross_entropy(p,t);new=stable_dice_bce(z,y)
  go=torch.autograd.grad(old,z,retain_graph=True)[0].abs().sum(1);gn=torch.autograd.grad(new,z)[0].abs().sum(1);mask=(y==1)&(z.argmax(1)!=1)
  n=int(mask.sum());wrong+=n;zero_old+=int(((go==0)&mask).sum());zero_new+=int(((gn==0)&mask).sum());old_grad+=float(go[mask].sum());new_grad+=float(gn[mask].sum());count+=1
net.load_state_dict(saved);assert all(torch.equal(v,net.state_dict()[k]) for k,v in saved.items())
atomic_json(r/f'{method}_s{seed}.json',{'status':'FINISHED','method':method,'seed':seed,'checks':checks,'domains':6,'sampled_train_slices':count,'wrong_foreground_pixels':wrong,'zero_gradient_old':zero_old,'zero_gradient_stable':zero_new,'sum_wrong_fg_abs_gradient_old':old_grad,'sum_wrong_fg_abs_gradient_stable':new_grad,'max_abs_logit':max_logit,'weights_buffers_restored':True,'no_training':True,'test_accessed':False})
print(json.dumps({'event':'finished','method':method,'seed':seed,'wrong_fg_zero_gradient_fraction_old':zero_old/max(1,wrong),'stable':zero_new/max(1,wrong),'max_abs_logit':max_logit}),flush=True)
