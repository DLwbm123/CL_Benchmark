import os,json,random,torch,cv2,numpy as np
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset,DataLoader
from datasets.seq_prostate import ProstateCL_Dataset
from datasets.seq_prostate import SequentialProstate
from models import get_model
from models.utils.medical import binary_targets

torch.set_num_threads(2)
if not os.environ.get('CL_DATA_ROOT'):raise RuntimeError('missing data root')
method=os.environ['ARM_METHOD']; seed=int(os.environ['ARM_SEED']); mode=os.environ['ARM_MODE']; result_path=Path(os.environ['ARM_RESULT'])
if method not in ('s3r','ted') or seed not in (1,2) or mode not in ('original','threshold'):raise ValueError('invalid arm')
ck=Path(os.environ['RUN_ROOT'])/f'{method}_s{seed}/checkpoints/T0_latest.pth'
c=torch.load(ck,map_location='cpu',weights_only=False)
args=Namespace(**c['config']);args.GPU_ids='0';args.test_only=False
random.seed(1106);np.random.seed(1106);torch.manual_seed(1106);torch.cuda.manual_seed_all(1106)
model=get_model(args,SequentialProstate.get_backbone(args),SequentialProstate.get_loss(),None).cuda()
model.load_network(str(ck),model.net)
if model.current_task!=1 or model.task_open:raise RuntimeError('unexpected checkpoint task state')
if method=='ted' and model.teacher is None:raise RuntimeError('missing teacher')
if method=='s3r' and (not model.anchor or not model.importance):raise RuntimeError('missing importance or anchor')
class TrainDataset(Dataset):
 def __init__(self):self.ds=ProstateCL_Dataset('2','train',256)
 def __len__(self):return len(self.ds)
 def __getitem__(self,i):
  x,y=self.ds[i]
  if mode=='threshold':
   raw=self.ds.labels_all[:,:,i]
   if not np.isfinite(raw).all() or np.any(raw<0) or np.any(raw>1):raise ValueError('invalid mask')
   y=torch.from_numpy(cv2.resize((raw>=.5).astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST)).long()
  return x,y
class Context:
 SETTING='domain-il'
ctx=Context();ctx.train_loader=DataLoader(TrainDataset(),batch_size=8,shuffle=True,num_workers=0)
model.begin_task(ctx)
for epoch in range(20):
 model.begin_epoch(epoch,20)
 for x,y in ctx.train_loader:
  loss=model.observe(x.cuda(),y.cuda())
  if not np.isfinite(loss) or loss>10000:raise FloatingPointError('invalid training loss')
 if not all(torch.isfinite(p).all() for p in model.net.parameters()):raise FloatingPointError('invalid parameters')
 print('epoch',epoch+1,flush=True)
model.eval();res={}
for site in (1,2):
 ds=ProstateCL_Dataset(str(site),'val',256);stats={'original':[0,0,0],'threshold':[0,0,0]};total=0
 with torch.inference_mode():
  for start in range(0,len(ds),8):
   batch=[ds[i] for i in range(start,min(start+8,len(ds)))];x=torch.stack([v[0] for v in batch]);orig=torch.stack([v[1] for v in batch]);raw=np.stack([ds.labels_all[:,:,i] for i in range(start,min(start+8,len(ds)))])
   thr=torch.from_numpy((raw>=.5).astype(np.int64));logits=model.net(x.cuda())
   if not torch.isfinite(logits).all():raise FloatingPointError('invalid logits')
   z=logits.argmax(1).cpu();total+=z.numel()
   for key,y in (('original',orig),('threshold',thr)):
    a=stats[key];a[0]+=int(((z==1)&(y==1)).sum());a[1]+=int((z==1).sum());a[2]+=int((y==1).sum())
 res[str(site)]={k:{'foreground_micro_dice':2*a[0]/(a[1]+a[2]) if a[1]+a[2] else None,'prediction_fraction':a[1]/total,'reference_fraction':a[2]/total} for k,a in stats.items()}
for state in model.opt.state.values():
 for v in state.values():
  if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():raise FloatingPointError('optimizer nonfinite')
result_path.write_text(json.dumps({'method':method,'seed':seed,'mode':mode,'epochs':20,'results':res},indent=2,allow_nan=False)+'\n')
