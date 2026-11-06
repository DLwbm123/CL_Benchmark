import json, os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from models.utils.shape import ShapeResUnet
from datasets.seq_prostate import ProstateCL_Dataset

torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(.20)
root=Path(os.environ['RUN_ROOT'])
out=Path(os.environ['WORK_ROOT'])
if not os.environ.get('CL_DATA_ROOT'): raise RuntimeError('missing data root')
rows={}
for method in ('s3r','ted'):
 rows[method]={}
 for seed in range(1,6):
  rows[method][str(seed)]={}
  for stage in range(6):
   p=root/f'{method}_s{seed}/checkpoints/T{stage}_latest.pth'
   c=torch.load(p,map_location='cpu',weights_only=False)
   if c['method'].lower()!=method: raise RuntimeError('method mismatch')
   net=resunet32('mid')
   if method=='s3r': net=ShapeResUnet(net,.5)
   weights={k[4:]:v for k,v in c['model'].items() if k.startswith('net.')}
   net.load_state_dict(weights,strict=True)
   if not all(torch.isfinite(x).all() for x in net.state_dict().values() if torch.is_tensor(x) and x.is_floating_point()): raise RuntimeError('nonfinite checkpoint')
   net=net.cuda().eval(); sites={}
   for site in (1,):
    ds=ProstateCL_Dataset(str(site),'val',256)
    tp=pred=actual=total=0
    with torch.inference_mode():
     for x,y in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):
      logits=net(x.cuda())
      if not torch.isfinite(logits).all(): raise RuntimeError('nonfinite logits')
      z=logits.argmax(1).cpu(); tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());actual+=int((y==1).sum());total+=y.numel()
    sites[str(site)]={'foreground_micro_dice':2*tp/(pred+actual) if pred+actual else None,'prediction_fraction':pred/total,'reference_fraction':actual/total}
   rows[method][str(seed)][str(stage)]=sites
   print('complete',method,seed,stage,flush=True)
   del net,c
   torch.cuda.empty_cache()

(out/'result.json').write_text(json.dumps({'split':'validation','checkpoint':'all six tasks, diagnostic only','results':rows},indent=2,allow_nan=False)+'\n')
