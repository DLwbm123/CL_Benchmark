import json,os,math,statistics
from pathlib import Path
import torch
root=Path(os.environ['WORK_ROOT']);state=json.loads((root/'state.json').read_text());assert state['status']=='FINISHED'
def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True
values=[];stages=[];means=[];counts=[]
for seed in (41,43):
 key=f'ted_s{seed}';r=json.loads((root/(key+'.json')).read_text());assert state['arms'][key]['returncode']==0 and r['tasks']==6 and r['epochs_per_domain']==50
 rows=[json.loads(l) for l in (root/(key+'.log')).read_text().splitlines() if l.startswith('{')]
 assert [(r['site'],r['epoch']) for r in rows if r.get('event')=='epoch']==[(s,e) for s in range(1,7) for e in range(1,51)]
 c=torch.load(root/(key+'_latest.pt'),map_location='cpu',weights_only=False);assert c['completed_site']==6 and finite(c['model']) and finite(c['optimizer']) and finite(c['teacher'])
 assert all(r['counts'][k]>0 for k in ['image_kd','ska','akt_gugf'])
 vv=list(r['results'].values());values+=vv;stages+=list(r['stage_scores'].values());means.append(statistics.mean(vv));counts.append(r['counts']);del c
out={'status':'VALIDATED','method':'ted_author_components','seeds':2,'domains':6,'epochs_per_domain':50,'mean_foreground_micro_dice':statistics.mean(values),'sd_across_seed_domain_means':statistics.stdev(means),'mean_just_learned_dice':statistics.mean(stages),'mean_change_final_minus_just_learned':statistics.mean(values)-statistics.mean(stages),'zero_seed_domain_combinations':sum(v==0 for v in values),'all_epoch_counts_rc_checkpoints_finite':True,'mechanism_counts_total':{k:sum(c[k] for c in counts) for k in counts[0]},'test_accessed':False,'allocated_bytes':state['allocated_bytes'],'elapsed_seconds_including_smoke':state['finished_at']-state['started_at'],'qualification':'author-component adapted validation experiment, not original-paper replication or independent final test'}
(root/'validated_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
