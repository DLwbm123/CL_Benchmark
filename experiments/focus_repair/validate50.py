import json,math,statistics,os
from pathlib import Path
import torch
root=Path(os.environ['WORK_ROOT'])
state=json.loads((root/'state.json').read_text());summary={'status':'VALIDATED','methods':{},'test_accessed':False,'scope':'completed method pair only; other arms may still run'}
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True
for method,epochs in [(os.environ.get('METHOD','focus'),50)]:
 vals=[];stages=[];seed_means=[];counts=[];selected=[]
 for seed in (41,43):
  key=f'{method}_s{seed}';r=json.loads((root/(key+'.json')).read_text())
  assert state['arms'][key]['returncode']==0 and r['status']=='FINISHED' and r['tasks']==6 and r['epochs_per_domain']==epochs
  logs=[json.loads(line) for line in (root/(key+'.log')).read_text().splitlines() if line.startswith('{')]
  actual=[(e['site'],e['epoch']) for e in logs if e.get('event')=='epoch']
  assert actual==[(s,e) for s in range(1,7) for e in range(1,epochs+1)]
  c=torch.load(root/(key+'_latest.pt'),map_location='cpu',weights_only=False)
  assert c['completed_site']==6 and finite(c['model']) and finite(c['optimizer'])
  if method=='focus':
   assert len(c['replay'])==5 and all(len(d)>=30 for d in c['replay']) and finite(c['replay'])
   assert r['counts']['replay_steps']>0 and r['counts']['gin_steps']>0
  elif method=='cauaug':
   assert all(len(v)==6 for v in c['archive'].values()) and finite(c['archive']) and finite(c['teacher'])
   assert r['counts']['cknet_steps']>0 and r['counts']['ftgen_steps']>0 and r['counts']['kd_steps']>0
  else:
   assert finite(c['teacher']) and all(r['counts'][k]>0 for k in ['image_kd','ska','akt_gugf'])
  values=list(r['results'].values());vals.extend(values);stages.extend(r['stage_scores'].values());seed_means.append(statistics.mean(values));counts.append(r['counts']);selected.extend(v['epoch'] for v in r['selection'].values());del c
 summary['methods'][method]={'seeds':2,'domains':6,'epochs_per_domain':epochs,'final_foreground_micro_dice_mean':statistics.mean(vals),'sd_across_seed_domain_means':statistics.stdev(seed_means),'mean_just_learned_domain_dice':statistics.mean(stages),'mean_change_final_minus_just_learned':statistics.mean(vals)-statistics.mean(stages),'zero_seed_domain_combinations':sum(v==0 for v in vals),'selection_at_last_epoch_count':sum(e==epochs for e in selected),'total_selection_count':len(selected),'mechanism_counts_total':{k:sum(c[k] for c in counts) for k in counts[0]},'all_epoch_counts_rc_checkpoints_finite':True}
(root/('validated_'+method+'.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
