import os,json,copy,torch
from pathlib import Path
from ted_stable_runner import make_model,objective,evaluate
from torch.utils.data import TensorDataset
from shared import finite
from numerics import numeric_checks
r=Path(os.environ['WORK_ROOT']);torch.set_num_threads(2);torch.manual_seed(63);checks=numeric_checks();net,seg,akt,agent=make_model();x=torch.rand(4,1,256,256,device='cuda');y=torch.zeros(4,256,256,dtype=torch.long,device='cuda');y[:,70:150,70:150]=1
for variant in ['stable_sup','stable_all']:
 os.environ['STABILITY_VARIANT']=variant;teacher=copy.deepcopy(net).requires_grad_(False).train();params={k:v.detach().clone() for k,v in teacher.named_parameters()};opt=torch.optim.Adam(net.parameters(),lr=.0002);opt.zero_grad(set_to_none=True);loss,_,dist=objective(net,teacher,x,y,seg,akt,agent);finite(loss,'smoke objective');loss.backward();opt.step();finite(net.state_dict(),'smoke model');finite(opt.state_dict(),'smoke optimizer');assert all(torch.equal(v,dict(teacher.named_parameters())[k]) for k,v in params.items())
 state={k:v.detach().clone() for k,v in net.state_dict().items()};evaluate(net,TensorDataset(x.cpu(),y.cpu()));assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items())
checks.update(status='PASSED',complete_TED_objective_backward_both_variants=True,teacher_parameters_unchanged=True,validation_state_restored=True);(r/'smoke.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
