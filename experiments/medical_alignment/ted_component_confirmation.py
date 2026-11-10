"""Fixed train-only audit of TED component balance; no optimizer updates."""
import os,json,copy,statistics
from pathlib import Path
import numpy as np
import torch
from ted_aligned import make_model,dataset,finite,atomic_json
from numerics import stable_dice_bce
from gradient_audit import measure
root=Path(os.environ['WORK_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT']);seed=int(os.environ['ARM_SEED']);arch=os.environ['ARM_BACKBONE']
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(seed);np.random.seed(seed)
net,_,akt,agent=make_model();teacher=copy.deepcopy(net).requires_grad_(False).train();old=base/f'ted_{arch}50';records=[]
def restore(model,state):
 with torch.no_grad():
  for k,v in model.named_buffers():v.copy_(state[k])
for site in range(2,7):
 net.load_state_dict(torch.load(old/f'ted_s{seed}_stage{site}.pt',map_location='cpu',weights_only=False)['model'])
 teacher.load_state_dict(torch.load(old/f'ted_s{seed}_stage{site-1}.pt',map_location='cpu',weights_only=False)['model'])
 net.train();teacher.train();ns={k:v.detach().clone() for k,v in net.state_dict().items()};ts={k:v.detach().clone() for k,v in teacher.state_dict().items()}
 ds=dataset(os.environ['DATA_ROOT'],site,'train',method='ted');used=set(np.linspace(0,len(ds)-1,16,dtype=int).tolist());remaining=[i for i in range(len(ds)) if i not in used];assert len(remaining)>=16;indices=np.array(remaining)[np.linspace(0,len(remaining)-1,16,dtype=int)];assert not set(indices.tolist()) & used
 for batch in range(4):
  torch.manual_seed(1200000+seed*100+site*10+batch);torch.cuda.manual_seed_all(1200000+seed*100+site*10+batch)
  pairs=[ds[int(i)] for i in indices[4*batch:4*(batch+1)]];x=torch.stack([v[0] for v in pairs]).cuda();y=torch.stack([v[1] for v in pairs]).cuda()
  logits=net(x);out=logits.softmax(1).clamp(1e-8,1-1e-8);sup=stable_dice_bce(logits,y)
  with torch.no_grad():target=teacher(x).softmax(1)
  noise=torch.randn_like(x);nout=net(noise).softmax(1).clamp(1e-8,1-1e-8)
  with torch.no_grad():ntarget=teacher(noise).softmax(1)
  comp={'image':agent.multi_class_cross_entropy_no_softmax(out,target),'noise':agent.multi_class_cross_entropy_no_softmax(nout,ntarget),'akt':akt(out,target)}
  g={k:measure(net,sup,v,.001) for k,v in comp.items()}
  rel={k:measure(net,comp['image'],comp[k],1.) for k in ('noise','akt')}
  total=measure(net,sup,sum(comp.values()),.001)
  records.append({'site':site,'batch':batch,'components':g,'relative_to_image':rel,'total':total})
  restore(net,ns);restore(teacher,ts)
 assert all(torch.equal(v,net.state_dict()[k]) for k,v in ns.items()) and all(torch.equal(v,teacher.state_dict()[k]) for k,v in ts.items())
 assert all(p.grad is None for p in net.parameters())
 del ds,ns,ts
assert len(records)==20;finite(records,'component probe')
summary={'status':'FINISHED','backbone':arch,'seed':seed,'samples':20,'train_slices':80,'states_restored':True,'no_parameter_updates':True,'test_accessed':False,'components':{},'records':records}
for k in ('image','noise','akt'):
 s={'weighted_to_supervised_median':statistics.median(r['components'][k]['weighted_ratio'] for r in records),'cosine_to_supervised_median':statistics.median(r['components'][k]['cosine'] for r in records)}
 if k!='image':s.update(norm_to_image_median=statistics.median(r['relative_to_image'][k]['weighted_ratio'] for r in records),cosine_to_image_median=statistics.median(r['relative_to_image'][k]['cosine'] for r in records))
 summary['components'][k]=s
atomic_json(root/f'{arch}_s{seed}.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
