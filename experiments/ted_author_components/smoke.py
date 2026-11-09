import copy,json,os,time
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from runner import make_model,objective,evaluate
from shared import finite,dataset
root=Path(os.environ['WORK_ROOT']);torch.set_num_threads(2);torch.manual_seed(975)
net,seg,akt,agent=make_model();teacher=copy.deepcopy(net).requires_grad_(False).train()
x=torch.rand(4,1,256,256,device='cuda').clamp(.001,.99);y=torch.zeros(4,256,256,dtype=torch.long,device='cuda');y[:,60:130,80:150]=1
params={k:p.clone() for k,p in teacher.named_parameters()};opt=torch.optim.Adam(net.parameters(),lr=2e-4)
loss,supervised,distill=objective(net,teacher,x,y,seg,akt,agent);finite(loss,'loss');finite(distill,'distillation');loss.backward();opt.step()
assert all(p.grad is None for p in teacher.parameters());assert all(torch.equal(v,dict(teacher.named_parameters())[k]) for k,v in params.items())
finite(net.state_dict(),'updated model');finite(opt.state_dict(),'updated optimizer')
state={k:v.clone() for k,v in net.state_dict().items()};ds=TensorDataset(x.cpu(),y.cpu());score=evaluate(net,ds)
assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
net.train();tp=pred=ref=0
with torch.no_grad():
 for a,b in ds:
  z=net(a[None].cuda()).argmax(1).cpu()[0]
  tp+=int(((z==1)&(b==1)).sum());pred+=int((z==1).sum());ref+=int((b==1).sum())
assert abs(score-2*tp/(pred+ref))<1e-12
real=dataset(os.environ['DATA_ROOT'],1,'train',method='ted');a,b=real[0]
assert a.min()>=.001-1e-7 and a.max()<=.99+1e-7 and set(b.unique().tolist())<={0,1}
r={'status':'PASSED','checks':['author UNet and DiceBCE forward/backward','image KD + Gaussian SKA + original broadcast AKT/GUGF finite backward','teacher parameters frozen','source train-BN slice inference equivalence','validation restores buffers','HDF5 integer-label and slice-minmax-clipping interface'],'test_accessed':False}
(root/'smoke.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
