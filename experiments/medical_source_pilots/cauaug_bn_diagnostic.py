"""Fixed-checkpoint validation diagnosis; no optimizer, test access, or model selection."""
import json,os,random,statistics,copy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from runner_v1 import make_model,dataset,finite,atomic_json
root=Path(os.environ['WORK_ROOT']);old=Path(os.environ['ORIGIN_ROOT'])
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
summary={'protocol':'same fixed final checkpoints, same validation blocks and CKNet random stream per seed/domain; factorial BN running vs batch statistics and batch size 4 vs 1; no checkpoint selection','methods':{},'test_accessed':False}
for mode,batch in [('running',4),('batch',4),('running',1),('batch',1)]:
 all_scores=[];pred_fracs=[];seed_scores=[]
 for seed in (41,43):
  net,_,agent=make_model('cauaug',os.environ['REFERENCE_ROOT'])
  ck=torch.load(old/f'cauaug_s{seed}_latest.pt',map_location='cpu',weights_only=False)
  scores=[]
  for site in range(1,7):
   net.load_state_dict(ck['model']);net.train(mode=='batch')
   ds=dataset(os.environ['DATA_ROOT'],site,'val','holdout',method='cauaug')
   tp=pred=ref=total=0
   with torch.random.fork_rng(devices=[0]),torch.no_grad():
    torch.manual_seed(100000+seed*10+site);torch.cuda.manual_seed_all(100000+seed*10+site)
    for x,y in DataLoader(ds,batch_size=batch,shuffle=False):
     aug=agent.cross_kernel_network(x.cuda());out=net(aug);finite(out,'logits');z=out.argmax(1).cpu()
     tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum());total+=y.numel()
   scores.append(2*tp/(pred+ref));pred_fracs.append(pred/total)
  seed_scores.append(statistics.mean(scores));all_scores.extend(scores)
  del net,ck;torch.cuda.empty_cache()
 summary['methods'][f'{mode}_b{batch}']={'mean_foreground_micro_dice':statistics.mean(all_scores),'sd_across_seed_domain_means':statistics.stdev(seed_scores),'mean_predicted_foreground_fraction':statistics.mean(pred_fracs),'zero_combinations':sum(s==0 for s in all_scores)}
atomic_json(root/'bn_diagnostic.json',summary);print(json.dumps(summary,indent=2))
