"""Paired five-seed confirmation from identical preserved first-stage models."""
import os,json,statistics,math
from pathlib import Path
import torch
base=Path(os.environ['WORK_ROOT']);roots={k:base/('ted_anchor_'+k) for k in ('control','half')}
plan=json.loads((base/'aggregation_protocol.json').read_text());seeds=plan['seeds'];summaries={};protocols={}
for label,root in roots.items():
 s=json.loads((root/'validated_ted.json').read_text());assert s['status']=='VALIDATED' and s['methods']['ted']['seeds']==5;summaries[label]=s['methods']['ted'];protocols[label]=json.loads((root/'protocol.json').read_text())
assert protocols['control']['later_domain_lr']==1e-4 and protocols['half']['later_domain_lr']==5e-5
for k in ('seeds','initialization','stream_seed','trained_domains','reused_domain','cudnn_deterministic','cudnn_benchmark','backbone','expected_parameters','epochs_per_domain','batch_size','gradient_epochs','stream_order','data','selection','supervision','distill_weight','teacher','evaluation','input'):
 assert protocols['control'][k]==protocols['half'][k],k
changes=[];learned={k:[] for k in roots}
for seed in seeds:
 cps={k:torch.load(r/f'ted_s{seed}_stage1.pt',map_location='cpu',weights_only=False) for k,r in roots.items()}
 source=Path(plan['methods']['ted'][str(seed)]);original=torch.load(source/f'ted_s{seed}_stage1.pt',map_location='cpu',weights_only=False)
 for cp in cps.values():
  assert cp['reused_anchor'] and cp['selection']==original['selection'] and cp['model'].keys()==original['model'].keys()
  assert all(torch.equal(v,original['model'][k]) for k,v in cp['model'].items())
 old=json.loads((source/f'ted_s{seed}.json').read_text());results={k:json.loads((r/f'ted_s{seed}.json').read_text()) for k,r in roots.items()}
 for label,r in results.items():
  assert r['stage_scores']['1']==old['stage_scores']['1'] and r['selection']['1']==old['selection']['1']
  learned[label].extend(r['stage_scores'][str(i)] for i in range(2,7))
 changes.append(statistics.mean(results['half']['results'].values())-statistics.mean(results['control']['results'].values()))
 del cps,original
mean=statistics.mean(changes);ld=statistics.mean(learned['half'])-statistics.mean(learned['control']);pos=sum(d>0 for d in changes)
assert math.isfinite(mean) and abs(mean-(summaries['half']['final_foreground_micro_dice_mean']-summaries['control']['final_foreground_micro_dice_mean']))<1e-12
out={'status':'VALIDATED_COMPLETE','seeds':seeds,'control':summaries['control'],'half':summaries['half'],'initial_models_selections_scores_identical_and_match_original_baseline':True,'same_continuation_rng_policy':True,'trained_domains':[2,3,4,5,6],'reused_domain':1,'mean_paired_delta':mean,'sample_sd_paired_deltas':statistics.stdev(changes),'improved_seed_count':pos,'worsened_seed_count':sum(d<0 for d in changes),'newly_learned_domains2_to6_mean_delta':ld,'bwt_delta':summaries['half']['bwt_previous_domains_mean']-summaries['control']['bwt_previous_domains_mean'],'prospective_confirmation_gate_passed':mean>=.02 and pos>=4 and ld>=-.02,'test_accessed':False,'scope':'conditional continuation from original shared stage1 checkpoints; not a new independent full six-domain-from-scratch cohort; no claim of whole-run bitwise determinism or statistical significance'}
(base/'comparison_ted_anchor.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2,allow_nan=False))
