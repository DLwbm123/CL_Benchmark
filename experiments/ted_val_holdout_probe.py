import os,json,random,torch,numpy as np,math
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset,DataLoader,Subset
from datasets.seq_prostate import ProstateCL_Dataset,SequentialProstate
from models import get_model

torch.set_num_threads(2)
if not os.environ.get('CL_DATA_ROOT'):raise RuntimeError('missing data root')
method=os.environ['ARM_METHOD'];seed=int(os.environ['ARM_SEED']);mode=os.environ['ARM_MODE'];output=Path(os.environ['ARM_RESULT'])
if method!='ted' or seed not in (17,19) or mode not in ('last','best'):raise ValueError('invalid arm')
p=Path(os.environ['RUN_ROOT'])/f'{method}_s1/checkpoints/T0_latest.pth'
c=torch.load(p,map_location='cpu',weights_only=False);args=Namespace(**c['config']);del c
args.GPU_ids='0';args.seed=seed;args.test_only=False
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
model=get_model(args,SequentialProstate.get_backbone(args),SequentialProstate.get_loss(),None).cuda()
class SliceDataset(Dataset):
 def __init__(self,site,split): self.base=ProstateCL_Dataset(str(site),split,256)
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x,y=self.base[i]
  if not torch.isfinite(x).all():raise FloatingPointError('nonfinite input')
  return x,y
"""Track the earliest maximum validation score using complete model state."""

class BestState:
 def __init__(self):
  self.best=float('-inf');self.epoch=None;self.state=None
 def consider(self,score,epoch,model):
  if not math.isfinite(score):raise FloatingPointError('nonfinite validation score')
  if not isinstance(epoch,int) or epoch<1:raise ValueError('invalid epoch')
  if score>self.best:
   state=model.state_dict()
   if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point()):raise FloatingPointError('nonfinite model state')
   self.state={k:v.detach().cpu().clone() for k,v in state.items()};self.best=score;self.epoch=epoch
 def restore(self,model):
  if self.state is None:raise RuntimeError('no valid validation scores')
  model.load_state_dict(self.state,strict=True)

def validation_block(site,block):
 ds=SliceDataset(site,'val');ends=ds.base.patient_info;n=len(ds)
 if len(ends)!=2 or int(ends[-1])!=n-1 or not 0<=int(ends[0])<int(ends[1]):raise ValueError('invalid validation boundaries')
 boundary=int(ends[0])+1
 if block=='selection':return Subset(ds,range(0,boundary))
 if block=='holdout':return Subset(ds,range(boundary,n))
 raise ValueError('invalid block')

class Context: SETTING='domain-il'
ctx=Context();selection={}
for task in (1,2):
 ctx.train_loader=DataLoader(SliceDataset(task,'train'),batch_size=8,shuffle=True,num_workers=0)
 model.begin_task(ctx);tracker=BestState() if mode=='best' else None
 for epoch in range(20):
  model.begin_epoch(epoch,20)
  for x,y in ctx.train_loader:
   loss=model.observe(x.cuda(),y.cuda())
   if not np.isfinite(loss) or loss>10000:raise FloatingPointError('invalid training loss')
  if not all(torch.isfinite(p).all() for p in model.net.parameters()):raise FloatingPointError('invalid parameters')
  if tracker is not None:
   model.net.eval();tp=pred=ref=0
   with torch.inference_mode():
    loader=DataLoader(validation_block(task,'selection'),batch_size=8,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(1))
    for vx,vy in loader:
     logits=model.net(vx.cuda())
     if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite validation logits')
     z=logits.argmax(1).cpu();tp+=int(((z==1)&(vy==1)).sum());pred+=int((z==1).sum());ref+=int((vy==1).sum())
   if pred+ref==0:raise ValueError('undefined validation dice')
   tracker.consider(2*tp/(pred+ref),epoch+1,model.net);model.net.train()
  print('task',task,'epoch',epoch+1,flush=True)
 if tracker is not None:
  tracker.restore(model.net);selection[str(task)]={'epoch':tracker.epoch,'score':tracker.best}
 model.end_task(ctx)
 if model.current_task!=task or model.task_open:raise RuntimeError('task state')
 if not all(torch.isfinite(p).all() for p in model.net.parameters()):raise FloatingPointError('invalid parameters after task')
 for state in model.opt.state.values():
  for v in state.values():
   if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():raise FloatingPointError('optimizer nonfinite')
model.eval();results={}
for site in (1,2):
 ds=validation_block(site,'holdout');tp=pred=ref=total=0
 with torch.inference_mode():
  for x,y in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):
   logits=model.net(x.cuda())
   if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite logits')
   z=logits.argmax(1).cpu();tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum());total+=y.numel()
 results[str(site)]={'foreground_micro_dice':2*tp/(pred+ref) if pred+ref else None,'prediction_fraction':pred/total,'reference_fraction':ref/total}
# Aggregate across sites/seeds in a privacy-preserving environment before publication.
output.write_text(json.dumps({'method':method,'seed':seed,'mode':mode,'results':results},allow_nan=False)+'\n')
