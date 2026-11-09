import os,json,copy,torch
from pathlib import Path
from torch.utils.data import TensorDataset
from base_network import resunet32
from runner import make_model,dc_ce,dice_bce,probabilities,evaluate,finite
from ted_runner import make_model as make_ted,objective as ted_objective,evaluate as ted_evaluate
root=Path(os.environ['WORK_ROOT']);torch.set_num_threads(2);torch.manual_seed(432)
x=torch.rand(4,1,256,256,device='cuda');y=torch.zeros(4,256,256,dtype=torch.long,device='cuda');y[:,60:150,60:150]=1
checks={};parameters={}
net,gin,_=make_model('focus',os.environ['REFERENCE_ROOT']);base=resunet32(size='mid').cuda();base.load_state_dict(net.state_dict());base.eval();net.eval()
with torch.no_grad():
 assert torch.allclose(base(x),net(x),rtol=1e-5,atol=1e-5),'unmodified-backbone equivalence'
 first=net.first(x);expected=net(x);net.freeze_layers('encoder');assert torch.allclose(expected,net.feature_forward([first]),rtol=1e-5,atol=1e-5)
enc={name:copy.deepcopy(module.state_dict()) for name,module in zip(range(6),net.encoder_modules())}
before=net.last.weight.detach().clone();net.train();opt=torch.optim.SGD([p for p in net.parameters() if p.requires_grad],lr=.01)
loss=dc_ce(net(gin()(x)),y);finite(loss,'focus loss');loss.backward();opt.step()
assert all(torch.equal(v,module.state_dict()[k]) for name,module in zip(range(6),net.encoder_modules()) for k,v in enc[name].items())
assert not torch.equal(before,net.last.weight)
parameters['focus']=sum(p.numel() for p in net.parameters());checks['focus']=['original ResUNet eval equivalence','first-feature replay equivalence','encoder params/buffers frozen','decoder updates with source CCD/GIN']
del net,base,opt;torch.cuda.empty_cache()
net,_,agent=make_model('cauaug',os.environ['REFERENCE_ROOT']);opt=torch.optim.Adam(net.parameters(),lr=.0002)
aug=agent.cross_kernel_network(x);loss=dice_bce(probabilities(net(aug)),y);finite(loss,'cauaug loss');loss.backward();opt.step()
state={k:v.clone() for k,v in net.state_dict().items()};evaluate(net,TensorDataset(x.cpu(),y.cpu()),'cauaug',agent)
assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items());parameters['cauaug']=sum(p.numel() for p in net.parameters());checks['cauaug']=['shared backbone source CKNet/DiceBCE backward','batch1 BN evaluation does not mutate state']
del net,opt,state;torch.cuda.empty_cache()
net,seg,akt,agent=make_ted();teacher=copy.deepcopy(net).requires_grad_(False).train();opt=torch.optim.Adam(net.parameters(),lr=.0002)
loss,_,dist=ted_objective(net,teacher,x,y,seg,akt,agent);finite(loss,'ted loss');finite(dist,'ted distill');loss.backward();opt.step()
assert all(p.grad is None for p in teacher.parameters());parameters['ted']=sum(p.numel() for p in net.parameters());checks['ted']=['shared backbone imageKD/SKA/AKT/GUGF backward','teacher params frozen']
assert len(set(parameters.values()))==1
r={'status':'PASSED','checks':checks,'parameter_counts':parameters,'note':'same ResUNet32-mid modules/parameterization; FOCUS encoder dropout and replay routing applied without changing convolutional modules'}
(root/'smoke.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
