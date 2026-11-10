"""Compare all five fixed seeds with the preserved TED U-Net control."""
import os,json,statistics,math
from pathlib import Path
root=Path(os.environ['WORK_ROOT']);window=root.parent
candidate=json.loads((root/'validated_ted.json').read_text())
baseline=json.loads((window/'validated_five_ted.json').read_text())
plan=json.loads((window/'aggregation_protocol.json').read_text())
protocol=json.loads((root/'protocol.json').read_text());seeds=plan['seeds']
assert protocol['seeds']==seeds and len(seeds)==5 and protocol['later_domain_lr']==5e-5
assert candidate['status']==baseline['status']=='VALIDATED'
a=baseline['methods']['ted'];b=candidate['methods']['ted'];assert a['seeds']==b['seeds']==5
changes=[]
for seed in seeds:
 old=json.loads((Path(plan['methods']['ted'][str(seed)])/f'ted_s{seed}.json').read_text())
 new=json.loads((root/f'ted_s{seed}.json').read_text())
 assert old['status']==new['status']=='FINISHED' and len(old['results'])==len(new['results'])==6
 changes.append(statistics.mean(new['results'].values())-statistics.mean(old['results'].values()))
delta=statistics.mean(changes);assert math.isfinite(delta) and abs(delta-(b['final_foreground_micro_dice_mean']-a['final_foreground_micro_dice_mean']))<1e-12
learned_delta=b['mean_just_learned_domain_dice']-a['mean_just_learned_domain_dice'];positive=sum(d>0 for d in changes)
result={'status':'VALIDATED_COMPLETE','seeds':seeds,'baseline':a,'candidate':b,'mean_paired_delta':delta,'sample_sd_paired_deltas':statistics.stdev(changes),'improved_seed_count':positive,'unchanged_seed_count':sum(d==0 for d in changes),'worsened_seed_count':sum(d<0 for d in changes),'just_learned_mean_delta':learned_delta,'bwt_delta':b['bwt_previous_domains_mean']-a['bwt_previous_domains_mean'],'prospective_development_gate_passed':delta>=.02 and positive>=4 and learned_delta>=-.02,'gate':'complete5 seeds; final mean improves>=.02; >=4/5 seed means improve; just-learned mean decrease<=.02; not a significance test','test_accessed':False,'claim_scope':'exploratory development optimization; retain original baseline and disclose selection'}
(root/'comparison_ted_lr.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2,allow_nan=False))
