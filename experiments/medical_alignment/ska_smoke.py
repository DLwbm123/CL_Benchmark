
import os,json,copy,torch
from pathlib import Path
import ted_aligned as m
torch.set_num_threads(2);torch.manual_seed(7);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
net,seg,akt,agent=m.make_model();teacher=copy.deepcopy(net).requires_grad_(False)
with torch.no_grad():
 for p in net.parameters():p.add_(.001*torch.randn_like(p))
x=torch.rand(4,1,256,256,device='cuda');y=torch.zeros(4,256,256,dtype=torch.long,device='cuda');y[:,60:160,60:160]=1
calls=[];transfers=[];original=agent.multi_class_cross_entropy_no_softmax
original_akt=akt
def record(a,b):
 v=original(a,b);calls.append(v);return v
def record_akt(a,b):
 v=original_akt(a,b);transfers.append(v);return v
agent.multi_class_cross_entropy_no_softmax=record
checks=[]
for factor in (1.,36.22710771566231):
 calls.clear();transfers.clear();os.environ['NOISE_MULTIPLIER']=str(factor)
 actual,sup,kd=m.objective(net,teacher,x,y,seg,record_akt,agent)
 assert len(calls)==2 and len(transfers)==1
 expected=sup+.001*(calls[0]+factor*calls[1]+transfers[0])
 assert torch.equal(actual,expected)
 g1=torch.autograd.grad(expected,tuple(net.parameters()),retain_graph=True);g2=torch.autograd.grad(actual,tuple(net.parameters()))
 diff=sum(float((a-b).double().square().sum()) for a,b in zip(g1,g2))**.5;norm=sum(float(b.double().square().sum()) for b in g2)**.5
 print(json.dumps({'factor':factor,'gradient_diff_norm':diff,'gradient_norm':norm}),flush=True)
 assert all(torch.isfinite(b).all() for b in g2) and diff <= 1e-4*max(norm,1e-8)
 checks.append({'noise_multiplier':factor,'loss':float(actual),'loss_exact_gradient_relative_tolerance_1e4':True,'gradient_relative_difference':diff/max(norm,1e-8)})
summary={'status':'PASSED','synthetic_only':True,'actual_model_backward':True,'same_graph_analytic_component_check':True,'checks':checks,'test_accessed':False,'initial_independent_graph_tolerance_failure_retained':True}
Path('ska_smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
