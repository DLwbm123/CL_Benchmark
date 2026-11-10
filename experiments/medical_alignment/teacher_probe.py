"""Frozen train-only teacher batch-statistic consistency diagnostic."""
import os,json,copy,statistics,math
from pathlib import Path
import numpy as np
import torch
from cau_aligned import make_model,dataset,probabilities,finite,atomic_json
from numerics import stable_dice_bce
from gradient_audit import measure
r=Path(os.environ['WORK_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT']);seed=int(os.environ['ARM_SEED']);arch=os.environ['ARM_BACKBONE'];key=f'{arch}_s{seed}'
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(seed);np.random.seed(seed)
net,_,agent=make_model('cauaug',os.environ['REFERENCE_ROOT']);teacher=copy.deepcopy(net).requires_grad_(False).train();net.train()
old=base/f'cauaug_{arch}50';last=torch.load(old/f'cauaug_s{seed}_latest.pt',map_location='cpu',weights_only=False)
records=[];all_grads=[]
def restore_buffers(model,buffers):
 with torch.no_grad():
  for k,v in model.named_buffers():v.copy_(buffers[k])
def confusion(z,y):
 q=z.argmax(1);return [int(((q==1)&(y==1)).sum()),int((q==1).sum()),int((y==1).sum())]
def score(c):return 2*c[0]/(c[1]+c[2]) if c[1]+c[2] else None
for site in range(2,7):
 c=torch.load(old/f'cauaug_s{seed}_stage{site}.pt',map_location='cpu',weights_only=False);t=torch.load(old/f'cauaug_s{seed}_stage{site-1}.pt',map_location='cpu',weights_only=False)
 net.load_state_dict(c['model']);teacher.load_state_dict(t['model']);net.train();teacher.train()
 before_n={k:v.detach().clone() for k,v in net.state_dict().items()};before_t={k:v.detach().clone() for k,v in teacher.state_dict().items()}
 bn={k:v.detach().clone() for k,v in net.named_buffers()};bt={k:v.detach().clone() for k,v in teacher.named_buffers()}
 for k,values in last['archive'].items():setattr(agent,k,[v.cuda() if torch.is_tensor(v) else v for v in values[:site-1]])
 ds=dataset(os.environ['DATA_ROOT'],site,'train',method='cauaug');indices=np.linspace(0,len(ds)-1,16,dtype=int)
 counts={k:[0,0,0] for k in ['batch_image','slice_image','batch_fourier','slice_fourier']};disagreements=total=0;grads=[]
 for batch in range(4):
  torch.manual_seed(1000000+seed*100+site*10+batch);torch.cuda.manual_seed_all(1000000+seed*100+site*10+batch)
  pairs=[ds[int(i)] for i in indices[batch*4:(batch+1)*4]];x=torch.stack([p[0] for p in pairs]).cuda();y=torch.stack([p[1] for p in pairs]).cuda()
  with torch.no_grad():aug=agent.cross_kernel_network(x);dom=agent.fourier_transformer_generator(x)
  logits=net(aug);out=probabilities(logits);dom_out=probabilities(net(dom));sup=stable_dice_bce(logits,y)
  with torch.no_grad():
   b_img=probabilities(teacher(x));b_dom=probabilities(teacher(dom));restore_buffers(teacher,bt)
   s_img=torch.cat([probabilities(teacher(xx[None])) for xx in x]);s_dom=torch.cat([probabilities(teacher(xx[None])) for xx in dom]);restore_buffers(teacher,bt)
  kd_batch=agent.multi_class_cross_entropy_no_softmax(out,b_img)+agent.multi_class_cross_entropy_no_softmax(dom_out,b_dom)
  kd_slice=agent.multi_class_cross_entropy_no_softmax(out,s_img)+agent.multi_class_cross_entropy_no_softmax(dom_out,s_dom)
  gb=measure(net,sup,kd_batch,.001);gs=measure(net,sup,kd_slice,.001);grads.append({'batch':gb,'slice':gs});all_grads.append({'batch':gb,'slice':gs});restore_buffers(net,bn)
  for name,z in [('batch_image',b_img),('slice_image',s_img),('batch_fourier',b_dom),('slice_fourier',s_dom)]:
   cc=confusion(z,y);counts[name]=[a+b for a,b in zip(counts[name],cc)]
  disagreements+=int((b_img.argmax(1)!=s_img.argmax(1)).sum())+int((b_dom.argmax(1)!=s_dom.argmax(1)).sum());total+=2*y.numel()
 assert all(torch.equal(v,net.state_dict()[k]) for k,v in before_n.items()) and all(torch.equal(v,teacher.state_dict()[k]) for k,v in before_t.items())
 scores={k:score(v) for k,v in counts.items()};assert all(v is not None and math.isfinite(v) for v in scores.values())
 records.append({'transition':site,'scores':scores,'teacher_argmax_disagreement':disagreements/total,'gradients':grads})
 del c,t,ds,before_n,before_t
summary={'status':'FINISHED','backbone':arch,'seed':seed,'transitions':5,'sampled_current_train_slices':80,'teacher_output_comparisons':160,'no_training':True,'test_accessed':False,'states_restored':True,'mean_teacher_batch_dice':statistics.mean((d['scores']['batch_image']+d['scores']['batch_fourier'])/2 for d in records),'mean_teacher_slice_dice':statistics.mean((d['scores']['slice_image']+d['scores']['slice_fourier'])/2 for d in records),'median_cosine_improvement':statistics.median(g['slice']['cosine']-g['batch']['cosine'] for g in all_grads if g['slice']['cosine'] is not None and g['batch']['cosine'] is not None),'mean_argmax_disagreement':statistics.mean(d['teacher_argmax_disagreement'] for d in records),'records':records}
finite(summary,'probe');atomic_json(r/(key+'.json'),summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
