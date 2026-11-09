import os,json,random,torch,numpy as np
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset,DataLoader
from datasets.seq_prostate import ProstateCL_Dataset,SequentialProstate
from models import get_model
import models.s3r as s3r_module

torch.set_num_threads(2)
if not os.environ.get('CL_DATA_ROOT'):raise RuntimeError('CL_DATA_ROOT required')
torch.cuda.set_per_process_memory_fraction(.24)
method=os.environ['ARM_METHOD'];seed=int(os.environ['ARM_SEED']);mode=os.environ['ARM_MODE'];output=Path(os.environ['ARM_RESULT'])
if method not in ('s3r','ted') or seed not in (11,13) or mode not in ('corrected','segonly'):raise ValueError('invalid arm')
if mode in ('corrected','segonly'):
 def corrected(mc_probabilities):
  avg=mc_probabilities.mean(0); confidence=torch.exp((avg*(avg+1e-6).log()).sum(1,keepdim=True))
  low=confidence.amin(dim=(-2,-1),keepdim=True); span=confidence.amax(dim=(-2,-1),keepdim=True)-low
  return torch.where(span>0,(confidence-low)/span.clamp_min(1e-12),torch.ones_like(confidence))
 s3r_module.confidence_mask=corrected
p=Path(os.environ['REFERENCE_CONFIG_CHECKPOINT']) # configuration only; no pretrained weights loaded
c=torch.load(p,map_location='cpu',weights_only=False);args=Namespace(**c['config']);del c
args.GPU_ids='0';args.seed=seed;args.test_only=False
site=int(os.environ['ARM_SITE'])
if site not in (5,6): raise ValueError('site')
if mode=='segonly': args.s3r_beta=0.;args.s3r_beta1=0.;args.s3r_lambda=0.
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
model=get_model(args,SequentialProstate.get_backbone(args),SequentialProstate.get_loss(),None).cuda()
class SliceDataset(Dataset):
 def __init__(self,site,split): self.base=ProstateCL_Dataset(str(site),split,256)
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x,y=self.base[i]
  if not torch.isfinite(x).all():raise FloatingPointError('nonfinite input')
  if False:
   lo=x.amin();span=x.amax()-lo
   x=torch.zeros_like(x) if span<=0 else (x-lo)/span
   x=x.clamp(.001,.99)
  return x,y
class Context: SETTING='domain-il'
ctx=Context()
for task in (site,):
 ctx.train_loader=DataLoader(SliceDataset(task,'train'),batch_size=8,shuffle=True,num_workers=0)
 model.begin_task(ctx)
 for epoch in range(20):
  model.begin_epoch(epoch,20)
  for x,y in ctx.train_loader:
   loss=model.observe(x.cuda(),y.cuda())
   if not np.isfinite(loss) or loss>10000:raise FloatingPointError('invalid training loss')
  if not all(torch.isfinite(p).all() for p in model.net.parameters()):raise FloatingPointError('invalid parameters')
  print('task',task,'epoch',epoch+1,flush=True)
 model.end_task(ctx)
 if model.current_task!=1 or model.task_open:raise RuntimeError('task state')
 if not all(torch.isfinite(p).all() for p in model.net.parameters()):raise FloatingPointError('invalid parameters after task')
 for state in model.opt.state.values():
  for v in state.values():
   if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():raise FloatingPointError('optimizer nonfinite')
model.eval();results={}
for site in (site,):
 ds=SliceDataset(site,'val');tp=pred=ref=total=0
 with torch.inference_mode():
  for x,y in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):
   logits=model.net(x.cuda())
   if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite logits')
   z=logits.argmax(1).cpu();tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum());total+=y.numel()
 results[str(site)]={'foreground_micro_dice':2*tp/(pred+ref) if pred+ref else None,'prediction_fraction':pred/total,'reference_fraction':ref/total}
output.write_text(json.dumps({'method':method,'seed':seed,'mode':mode,'tasks':1,'epochs':20,'results':results},indent=2,allow_nan=False)+'\n')
