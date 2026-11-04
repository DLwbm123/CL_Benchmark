import json,time,torch,os
from pathlib import Path
from torch.utils.data import DataLoader
from backbone.ResUnet import resunet32
from datasets.seq_prostate import ProstateCL_Dataset
from models.utils.smg_intensity import normalize_per_slice
torch.set_num_threads(2)
p=Path(os.environ['CHECKPOINT_DIR']); out=Path(os.environ['OUTPUT_DIR'])
result={}
for task in range(6):
 c=torch.load(p/f'T{task}_latest.pth',map_location='cpu',weights_only=True); net=resunet32('mid').cuda().eval(); net.load_state_dict({k.removeprefix('net.core.'):v for k,v in c['model'].items() if k.startswith('net.core.')}); result[str(task)]={}
 for site in range(1,min(task+2,3)):
  tp=pred=actual=total=0
  with torch.no_grad():
   for x,y in DataLoader(ProstateCL_Dataset(str(site),'val',256),batch_size=8):
    z=net(normalize_per_slice(x.cuda())).argmax(1).cpu(); tp+=int(((z==1)&(y==1)).sum()); pred+=int((z==1).sum()); actual+=int((y==1).sum()); total+=y.numel()
  result[str(task)][str(site)]={'dice':2*tp/(pred+actual) if pred+actual else None,'predicted_fraction':pred/total,'reference_fraction':actual/total}
 print(task,result[str(task)],flush=True)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
