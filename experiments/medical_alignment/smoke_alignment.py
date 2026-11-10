import os,json,torch,importlib
from pathlib import Path
from torch.utils.data import TensorDataset
from gradient_audit import measure
from numerics import numeric_checks
base=Path(os.environ['WORK_ROOT']);checks=numeric_checks();torch.set_num_threads(2)
# Measurements must not change parameters, RNG, or the eventual update gradient.
m=torch.nn.Linear(3,2).cuda();x=torch.randn(4,3,device='cuda');a=m(x);sup=a.square().mean();kd=(a-1).square().mean();expected=torch.autograd.grad(sup+.001*kd,list(m.parameters()),retain_graph=True)
rng=torch.get_rng_state();crng=torch.cuda.get_rng_state();before={k:v.clone() for k,v in m.state_dict().items()}
measure(m,sup,kd,.001);assert all(p.grad is None for p in m.parameters());(sup+.001*kd).backward()
assert all(torch.allclose(p.grad,g,atol=1e-7,rtol=1e-6) for p,g in zip(m.parameters(),expected));assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()) and torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
results={}
for method,module in [('ted','ted_aligned'),('cauaug','cau_aligned')]:
 for backbone in ['author','shared']:
  r=base/(method+'_'+backbone);r.mkdir(exist_ok=False);(r/'protocol.json').write_text(json.dumps({'seeds':[41],'epochs_per_domain':{method:1},'gradient_epochs':[1]}))
  os.environ.update(WORK_ROOT=str(r),ARM_METHOD=method,ARM_BACKBONE=backbone,ARM_SEED='41',STABILITY_VARIANT='stable_sup',DISTILL_WEIGHT='.001',DATA_ROOT='synthetic')
  mod=importlib.import_module(module);torch.manual_seed(63);xx=torch.rand(4,1,256,256);yy=torch.zeros(4,256,256,dtype=torch.long);yy[:,70:150,70:150]=1
  original=mod.dataset;mod.dataset=lambda *args,**kwargs:TensorDataset(xx.clone(),yy.clone())
  mod.run();mod.dataset=original
  d=json.loads((r/f'{method}_s41.json').read_text());c=torch.load(r/f'{method}_s41_latest.pt',map_location='cpu',weights_only=False);mod.finite(c,'checkpoint');assert d['status']=='FINISHED' and c['completed_site']==6
  history=json.loads((r/f'{method}_s41_stage_matrix.json').read_text());assert [len(history[str(i)]) for i in range(1,7)]==list(range(1,7));assert len(list(r.glob('*_stage[1-6].pt')))==6
  if method=='ted':assert all(d['counts'][k]==5 for k in ['image_kd','ska','akt_gugf'])
  else:assert d['counts']['cknet_steps']==6 and d['counts']['ftgen_steps']==5 and d['counts']['kd_steps']==5 and all(len(v)==6 for v in c['archive'].values())
  results[method+'_'+backbone]={'status':'PASSED','six_stage_pipeline':True,'all_mechanisms_active':True,'finite_checkpoint':True,'complete_stage_matrix':True}
  del c
checks.update(status='PASSED',gradient_measurement_inert=True,variants=results,synthetic_only=True)
(base/'smoke_summary.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
