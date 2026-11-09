import os,json,torch
from pathlib import Path
from torch.utils.data import TensorDataset,DataLoader
from runner_v2 import make_model,evaluate,finite
root=Path(os.environ['WORK_ROOT']);torch.set_num_threads(2);torch.manual_seed(913)
net,_,agent=make_model('cauaug',os.environ['REFERENCE_ROOT'])
x=torch.rand(3,1,256,256);y=torch.zeros(3,256,256,dtype=torch.long);y[:,80:160,80:160]=1;ds=TensorDataset(x,y)
state={k:v.clone() for k,v in net.state_dict().items()};rng=torch.get_rng_state().clone()
score=evaluate(net,ds,'cauaug',agent,site=1,seed=41)
assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items()),'evaluation changed model state'
assert torch.equal(rng,torch.get_rng_state()),'evaluation advanced caller RNG'
net.train();tp=pred=ref=0
with torch.random.fork_rng(devices=[0]),torch.no_grad():
 torch.manual_seed(100411);torch.cuda.manual_seed_all(100411)
 for a,b in DataLoader(ds,batch_size=1,shuffle=False):
  out=net(agent.cross_kernel_network(a.cuda()));finite(out,'source logits');z=out.argmax(1).cpu()
  tp+=int(((z==1)&(b==1)).sum());pred+=int((z==1).sum());ref+=int((b==1).sum())
assert abs(score-2*tp/(pred+ref))<1e-12,'source batch1 inference differs'
r={'status':'PASSED','checks':['source train-BN batch1 inference equivalence','model parameters and BN buffers unchanged by validation','caller RNG preserved'],'original_core_smoke':'passed unchanged source_components and training mechanisms in v1'}
(root/'inference_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
