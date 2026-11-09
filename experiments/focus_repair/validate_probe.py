import os,json,math,statistics
from pathlib import Path
import torch
root=Path(os.environ['WORK_ROOT']);state=json.loads((root/'state.json').read_text());p=json.loads((root/'protocol.json').read_text());assert state['status']=='FINISHED' and len(state['arms'])==16

def finite(v):
 if torch.is_tensor(v):return bool(torch.isfinite(v).all())
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,float):return math.isfinite(v)
 return True

records={};summary={'status':'VALIDATED','domains':2,'seeds':2,'epochs_per_arm':50,'arms':16,'variants':{},'test_accessed':False,'interpretation':'independent single-domain development diagnosis; not continual learning performance'}
for variant in ['full_sgd','no_ccd','no_gin','full_adam']:
 rr=[]
 for site in p['sites']:
  for seed in p['seeds']:
   key=f'{variant}_d{site}_s{seed}';r=json.loads((root/(key+'.json')).read_text());assert state['arms'][key]['returncode']==0 and r['epochs']==50 and not r['test_accessed'] and finite(r)
   assert [row['epoch'] for row in r['trace']]==list(range(1,51))
   cp=torch.load(root/(key+'_best.pt'),map_location='cpu',weights_only=False);assert finite(cp['model']) and finite(cp['optimizer']) and cp['epoch']==r['selection_epoch'];del cp
   rr.append(r);records[(variant,site,seed)]=r
 vals=[r['holdout_dice'] for r in rr];seedmeans=[statistics.mean(records[(variant,d,s)]['holdout_dice'] for d in p['sites']) for s in p['seeds']]
 summary['variants'][variant]={'mean_foreground_micro_dice':statistics.mean(vals),'sd_across_seed_domain_means':statistics.stdev(seedmeans),'zero_domain_seed_combinations':sum(v==0 for v in vals),'mean_clipped_step_fraction_first10epochs':statistics.mean(t['clipped_step_fraction'] for r in rr for t in r['trace'][:10]),'mean_preclip_gradient_norm_first10epochs':statistics.mean(t['gradient_norm_before_clip_mean'] for r in rr for t in r['trace'][:10]),'all_epochs_rc_checkpoint_finite':True}
for variant in ['no_ccd','no_gin','full_adam']:
 delta=[records[(variant,d,s)]['holdout_dice']-records[('full_sgd',d,s)]['holdout_dice'] for d in p['sites'] for s in p['seeds']]
 summary['variants'][variant]['paired_delta_vs_full_sgd_mean']=statistics.mean(delta);summary['variants'][variant]['improved_pairs_of_four']=sum(v>0 for v in delta)
a=summary['variants']['full_adam'];b=summary['variants']['full_sgd']
summary['full_adam_sequential_validation_gate']={'passed':a['improved_pairs_of_four']==4 and a['paired_delta_vs_full_sgd_mean']>=.05 and a['zero_domain_seed_combinations']==0 and a['mean_clipped_step_fraction_first10epochs']<b['mean_clipped_step_fraction_first10epochs'],'rule':'all4 paired improvements, mean>=.05, no zero, lower early clipping; gate fixed before complete results; does not establish sequential efficacy'}
(root/'validated_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
