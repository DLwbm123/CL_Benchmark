"""Independent single-domain mechanism diagnosis; not a continual-learning result."""
import os,json,time,copy,random,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from runner import make_model,dataset,dc_ce,evaluate,finite,atomic_json,atomic_checkpoint
root=Path(os.environ['WORK_ROOT']);seed=int(os.environ['ARM_SEED']);variant=os.environ['ARM_VARIANT'];site=int(os.environ['ARM_SITE']);key=f'{variant}_d{site}_s{seed}';p=json.loads((root/'protocol.json').read_text())
assert variant in ['full_sgd','no_ccd','no_gin','full_adam'] and site in p['sites'] and seed in p['seeds']
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
net,gin,_=make_model('focus',os.environ['REFERENCE_ROOT'])
if variant=='no_ccd':
 original=net.sparsify_features
 def consume_but_bypass(x,**kwargs):
  original(x,**kwargs)
  return x
 net.sparsify_features=consume_but_bypass
base_lr=.0002 if variant=='full_adam' else .01
opt=torch.optim.Adam(net.parameters(),lr=base_lr,weight_decay=3e-5) if variant=='full_adam' else torch.optim.SGD(net.parameters(),lr=base_lr,momentum=.99,nesterov=True,weight_decay=3e-5)
ds=dataset(os.environ['DATA_ROOT'],site,'train',method='focus');val=dataset(os.environ['DATA_ROOT'],site,'val','selection',method='focus');loader=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0)
best=-math.inf;best_state=best_opt=None;best_epoch=0;steps=0;gin_calls=0;trace=[]
for epoch in range(p['epochs']):
 start=time.time();net.train();loss_sum=norm_sum=0.;max_logit=0.;clipped=n=0
 for group in opt.param_groups:group['lr']=base_lr*(1-epoch/p['epochs'])**.9
 for x,y in loader:
  x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
  if random.random()<.5:
   with torch.no_grad():aug=gin()(x)
   if variant!='no_gin':x=aug
   gin_calls+=1
  logits=net(x);loss=dc_ce(logits,y);finite(loss,'loss');loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),12.,error_if_nonfinite=True);opt.step();loss_sum+=float(loss.detach());norm_sum+=float(gn);max_logit=max(max_logit,float(logits.detach().abs().max()));clipped+=int(gn>12);n+=1;steps+=1
 finite(net.state_dict(),'model');finite(opt.state_dict(),'optimizer');score=evaluate(net,val,'focus',site=site,seed=seed)
 if score>best:
  best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_opt=copy.deepcopy(opt.state_dict())
 row={'event':'epoch','epoch':epoch+1,'site':site,'loss':loss_sum/n,'gradient_norm_before_clip_mean':norm_sum/n,'clipped_step_fraction':clipped/n,'max_abs_logit':max_logit,'selection_dice':score,'best_epoch':best_epoch,'steps':steps,'seconds':time.time()-start};trace.append(row);print(json.dumps(row),flush=True);atomic_json(root/(key+'_progress.json'),row)
net.load_state_dict(best_state);score=evaluate(net,dataset(os.environ['DATA_ROOT'],site,'val','holdout',method='focus'),'focus',site=site,seed=seed)
atomic_checkpoint(root/(key+'_best.pt'),{'model':best_state,'optimizer':best_opt,'epoch':best_epoch,'seed':seed,'variant':variant,'site':site})
atomic_json(root/(key+'.json'),{'status':'FINISHED','variant':variant,'seed':seed,'site':site,'epochs':p['epochs'],'selection_epoch':best_epoch,'holdout_dice':score,'steps':steps,'gin_draws':gin_calls,'finite':True,'test_accessed':False,'trace':trace})
print(json.dumps({'event':'finished','key':key}),flush=True)
