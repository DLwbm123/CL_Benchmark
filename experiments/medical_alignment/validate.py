import json,math,statistics,os
import numpy as np
from pathlib import Path
import torch
root=Path(os.environ['WORK_ROOT']);protocol=json.loads((root/'protocol.json').read_text())
state=json.loads((root/'state.json').read_text());summary={'status':'VALIDATED','methods':{},'test_accessed':False,'scope':'completed method pair only; other arms may still run'}
def median_or_none(xs):
 xs=list(xs);return statistics.median(xs) if xs else None
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True
for method,epochs in [(os.environ['METHOD'],json.loads((root/'protocol.json').read_text())['epochs_per_domain'][os.environ['METHOD']])]:
 vals=[];stages=[];seed_means=[];counts=[];selected=[];audits=[];bwt=[];forget=[]
 for seed in (41,43):
  key=f'{method}_s{seed}';r=json.loads((root/(key+'.json')).read_text())
  assert state['arms'][key]['returncode']==0 and r['status']=='FINISHED' and r['tasks']==6 and r['epochs_per_domain']==epochs
  logs=[json.loads(line) for line in (root/(key+'.log')).read_text().splitlines() if line.startswith('{')]
  actual=[(e['site'],e['epoch']) for e in logs if e.get('event')=='epoch']
  ready=[e for e in logs if e.get('event')=='model_ready'];assert len(ready)==1 and ready[0]['parameters']==protocol['expected_parameters']
  g=[e for e in logs if e.get('event')=='gradient_audit'];assert [(e['site'],e['epoch']) for e in g]==[(i,j) for i in range(2,7) for j in protocol['gradient_epochs']] and finite(g);audits.extend(g)
  matrix=json.loads((root/(key+'_stage_matrix.json')).read_text());assert set(matrix)==set(map(str,range(1,7))) and finite(matrix)
  for task in range(1,7):
   assert set(matrix[str(task)])==set(map(str,range(1,task+1))) and abs(matrix[str(task)][str(task)]-r['stage_scores'][str(task)])<1e-8 and abs(matrix['6'][str(task)]-r['results'][str(task)])<1e-6
   stage_cp=torch.load(root/(key+f'_stage{task}.pt'),map_location='cpu',weights_only=False);assert stage_cp['completed_site']==task and finite(stage_cp['model']) and stage_cp['protocol'].get('noise_multiplier',1.)==protocol.get('noise_multiplier',1.);del stage_cp
   if task<6:
    bwt.append(matrix['6'][str(task)]-matrix[str(task)][str(task)])
    forget.append(max(matrix[str(t)][str(task)] for t in range(task,6))-matrix['6'][str(task)])
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
  assert set(r['results'])=={str(i) for i in range(1,7)} and finite(r['results']) and finite(r['stage_scores'])
  assert all(0<=v<=1 for v in r['results'].values())
  values=list(r['results'].values());vals.extend(values);stages.extend(r['stage_scores'].values());seed_means.append(statistics.mean(values));counts.append(r['counts']);selected.extend(v['epoch'] for v in r['selection'].values());del c
 summary['methods'][method]={'seeds':2,'domains':6,'epochs_per_domain':epochs,'final_foreground_micro_dice_mean':statistics.mean(vals),'sd_across_seed_domain_means':statistics.stdev(seed_means),'mean_just_learned_domain_dice':statistics.mean(stages),'mean_change_final_minus_just_learned':statistics.mean(vals)-statistics.mean(stages),'zero_seed_domain_combinations':sum(v==0 for v in vals),'selection_at_last_epoch_count':sum(e==epochs for e in selected),'total_selection_count':len(selected),'mechanism_counts_total':{k:sum(c[k] for c in counts) for k in counts[0]},'all_epoch_counts_rc_checkpoints_finite':True}
summary['methods'][method].update(backbone=protocol['backbone'],parameters=protocol['expected_parameters'],bwt_previous_domains_mean=statistics.mean(bwt),forgetting_from_best_observed_stage_mean=statistics.mean(forget),previous_domain_seed_combinations=len(bwt),gradient_audit_samples=len(audits),gradient_weighted_ratio_median=median_or_none(a['weighted_ratio'] for a in audits if a['weighted_ratio'] is not None),gradient_cosine_median=median_or_none(a['cosine'] for a in audits if a['cosine'] is not None),zero_distillation_gradient_samples=sum(a['distillation_norm']==0 for a in audits),stage_matrices_checkpoints_validated=True)
(root/('validated_'+method+'.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
