import os,json,copy,random,statistics,time
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
from runner import make_model,dataset,finite,atomic_json
root=Path(os.environ['WORK_ROOT']);seed=int(os.environ['ARM_SEED']);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
old=Path(os.environ['OLD_ROOT']);cp=torch.load(old/f'focus_s{seed}_latest.pt',map_location='cpu',weights_only=False)
net,_,_=make_model('focus',os.environ['REFERENCE_ROOT']);net.load_state_dict(cp['model']);net.freeze_layers('encoder');before={k:v.clone() for k,v in net.state_dict().items()};out={}
for mode,batch in [('running_b4',4),('running_b1',1),('slice_bn',1)]:
 scores=[];net.load_state_dict(before);net.eval()
 # Keep CCD/GIN disabled. Switch only BN modules to batch statistics.
 if mode=='slice_bn':
  for m in net.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.train()
 with torch.random.fork_rng(devices=[0]),torch.inference_mode():
  for site in range(1,7):
   ds=dataset(os.environ['DATA_ROOT'],site,'val','holdout',method='focus');tp=pred=ref=0
   for x,y in DataLoader(ds,batch_size=batch,shuffle=False,num_workers=0):
    z=net(x.cuda());finite(z,'diagnostic logits');z=z.argmax(1).cpu();tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum())
   scores.append(2*tp/(pred+ref))
 out[mode]={'domain_mean':statistics.mean(scores),'zero_domains':sum(v==0 for v in scores),'scores_private':scores}
 net.load_state_dict(before);assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
expected=json.loads((old/f'focus_s{seed}.json').read_text())['results'];assert max(abs(a-b) for a,b in zip(out['running_b4']['scores_private'],expected.values()))<1e-10
# Training-only, coarse aggregate input scale; never publish individual slices.
means=[];stds=[];lo=[];hi=[];zeros=[]
for site in range(1,7):
 ds=dataset(os.environ['DATA_ROOT'],site,'train',method='focus');x=ds.tensors[0]
 means.extend(x.mean((1,2,3)).tolist());stds.extend(x.std((1,2,3),unbiased=False).tolist());lo.append(float(x.min()));hi.append(float(x.max()));zeros.extend((x==0).float().mean((1,2,3)).tolist())
input_summary={'mean_slice_mean':statistics.mean(means),'mean_slice_sd':statistics.mean(stds),'global_min':min(lo),'global_max':max(hi),'mean_zero_fraction':statistics.mean(zeros)}
atomic_json(root/f'diagnostic_s{seed}.json',{'seed':seed,'status':'FINISHED','modes':out,'state_restored':True,'original_result_reproduced':True,'train_input_aggregate':input_summary,'no_training':True,'test_accessed':False})
print(json.dumps({'status':'FINISHED','seed':seed,'means':{k:v['domain_mean'] for k,v in out.items()},'input':input_summary}),flush=True)
