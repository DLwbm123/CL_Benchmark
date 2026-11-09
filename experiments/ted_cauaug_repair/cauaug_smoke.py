import os,json,torch
from pathlib import Path
from torch.utils.data import TensorDataset
import cauaug_stable_runner as m
from numerics import numeric_checks
r=Path(os.environ['WORK_ROOT']);checks=numeric_checks();torch.manual_seed(63)
x=torch.rand(4,1,256,256);y=torch.zeros(4,256,256,dtype=torch.long);y[:,70:150,70:150]=1
def data(*args,**kwargs):return TensorDataset(x.clone(),y.clone())
m.dataset=data;m.run()
a=json.loads((r/'cauaug_s41.json').read_text());c=torch.load(r/'cauaug_s41_latest.pt',map_location='cpu',weights_only=False)
assert a['status']=='FINISHED' and a['counts']['cknet_steps']==6 and a['counts']['ftgen_steps']==5 and a['counts']['kd_steps']==5
assert c['completed_site']==6 and all(len(v)==6 for v in c['archive'].values())
m.finite(c,'smoke checkpoint')
net,_,agent=m.make_model('cauaug',os.environ['REFERENCE_ROOT']);net.load_state_dict(c['model']);state={k:v.detach().clone() for k,v in net.state_dict().items()};rng=torch.get_rng_state();crng=torch.cuda.get_rng_state()
m.evaluate(net,data(),'cauaug',agent,1,41)
assert all(torch.equal(v,net.state_dict()[k]) for k,v in state.items()) and torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
checks.update(status='PASSED',synthetic_only=True,full_six_stage_pipeline=True,all_mechanisms_active=True,finite_model_optimizer_teacher_archive=True,validation_state_rng_restored=True)
(r/'smoke.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
