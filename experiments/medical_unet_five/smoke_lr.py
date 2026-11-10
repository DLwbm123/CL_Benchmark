
import pathlib,os,json,torch,contextlib
from torch.utils.data import TensorDataset
import ted_lr_aligned as m
root=pathlib.Path(os.environ['WORK_ROOT']);r=root/'synthetic_preflight';r.mkdir();protocol=json.loads((root/'protocol.json').read_text());protocol.update(seeds=[41],epochs_per_domain={'ted':1},gradient_epochs=[1]);(r/'protocol.json').write_text(json.dumps(protocol))
os.environ['WORK_ROOT']=str(r);torch.manual_seed(63);x=torch.rand(4,1,256,256);y=torch.zeros(4,256,256,dtype=torch.long);y[:,70:150,70:150]=1
m.dataset=lambda *args,**kwargs:TensorDataset(x.clone(),y.clone())
with (r/'smoke.log').open('w') as log,contextlib.redirect_stdout(log):m.run()
events=[json.loads(l) for l in (r/'smoke.log').read_text().splitlines() if l.startswith('{')];rates=[e['learning_rate_used'] for e in events if e['event']=='epoch'];assert rates==[2e-4]+[5e-5]*5
result=json.loads((r/'ted_s41.json').read_text());cp=torch.load(r/'ted_s41_latest.pt',map_location='cpu',weights_only=False);m.finite(cp,'checkpoint');assert result['status']=='FINISHED' and cp['completed_site']==6 and all(result['counts'][k]==5 for k in ('image_kd','ska','akt_gugf'))
summary={'status':'PASSED','synthetic_only':True,'six_stage_actual_model_backward':True,'observed_learning_rates':rates,'all_ted_components_active':True,'finite_checkpoint':True,'test_accessed':False};(root/'smoke_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
