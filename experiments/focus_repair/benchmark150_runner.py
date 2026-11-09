"""Six-domain validation pilots; author components with explicit HDF5 adaptation.
All patient-derived artifacts stay in the private experiment directory.
"""
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from types import MethodType
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from source_components import focus_components, cauaug_components
from shared_backbone import FocusResUNet
from base_network import resunet32

NAMES = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']


def finite(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(label)
    elif isinstance(value, dict):
        for v in value.values():
            finite(v, label)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v, label)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise FloatingPointError(label)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def atomic_checkpoint(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dataset(root, site, split, block=None, method=None):
    if split not in ('train', 'val'):
        raise ValueError('test access forbidden in this pilot')
    with h5py.File(Path(root)/(NAMES[site-1]+'.h5'), 'r') as f:
        x = np.asarray(f[split+'_images']).transpose(2, 0, 1).copy()
        y = np.asarray(f[split+'_labels']).transpose(2, 0, 1).copy()
        if split == 'val':
            ends = np.asarray(f['patient_info_val']).astype(int)
            if len(ends) < 2 or ends[-1] != len(x)-1 or not np.all(np.diff(ends)>0):
                raise ValueError('invalid sequence boundaries')
            boundary = int(ends[0])+1
            selected = slice(0,boundary) if block == 'selection' else slice(boundary,None)
            x,y = x[selected],y[selected]
    if x.shape != y.shape or x.shape[1:] != (256,256):
        raise ValueError('unexpected data shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.min()<0 or y.max()>1:
        raise ValueError('invalid images or labels')
    # Existing benchmark loader casts fractional HDF5 labels to integer.
    # Preserve that frozen label policy; do not introduce a new threshold here.
    y=y.astype(np.int64)
    if not np.isin(y,[0,1]).all():raise ValueError('invalid integer labels')
    # CauAug expects [0,1]. With no verified volume boundaries, explicitly
    # use slice-level minmax (an adaptation, NOT author case-level minmax).
    x=torch.from_numpy(x).float().unsqueeze(1)
    if method=='cauaug':
        low=x.amin((2,3),keepdim=True);span=x.amax((2,3),keepdim=True)-low
        x=torch.where(span>0,(x-low)/span.clamp_min(1e-12),torch.zeros_like(x))
    return TensorDataset(x, torch.from_numpy(y).long())


def primary(logits):
    return logits[0] if isinstance(logits, (tuple,list)) else logits


def dc_ce(outputs, target):
    outputs = outputs if isinstance(outputs,(tuple,list)) else [outputs]
    weights = [1/(2**i) if i < len(outputs)-1 or len(outputs)==1 else 0 for i in range(len(outputs))]
    total = outputs[0].new_zeros(())
    for out,w in zip(outputs,weights):
        if not w:
            continue
        y=F.interpolate(target[:,None].float(),size=out.shape[2:],mode='nearest')[:,0].long()
        prob=out.softmax(1)[:,1]
        truth=(y==1).float()
        # nnU-Net batch Dice, foreground only, smooth 1e-5.
        dice=(2*(prob*truth).sum()+1e-5)/(prob.sum()+truth.sum()+1e-5)
        total=total+(w/sum(weights))*(F.cross_entropy(out,y)-dice)
    return total


def dice_bce(prob, target):
    truth=F.one_hot(target,2).permute(0,3,1,2).float()
    dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
    return dice+F.binary_cross_entropy(prob,truth)


def probabilities(logits):
    return logits.softmax(1).clamp(1e-8,1-1e-8)


def evaluate(net, ds, method, agent=None, site=1, seed=41):
    previous_mode=net.training
    # Author CauAug predictor leaves the model in train mode and processes
    # one slice at a time. Batch-statistic inference is required for source parity.
    # Restore BN buffers afterwards so validation cannot change persisted state.
    bn_buffers={k:v.detach().clone() for k,v in net.named_buffers()} if method=='cauaug' else {}
    net.train(method=='cauaug')
    tp=pred=ref=0
    # Fixed evaluation random stream prevents random CKNet draws deciding epochs.
    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
        torch.manual_seed(100000+seed*10+site)
        torch.cuda.manual_seed_all(100000+seed*10+site)
        for x,y in DataLoader(ds,batch_size=1 if method=='cauaug' else 4,shuffle=False,num_workers=0):
            x=x.cuda()
            if method=='cauaug':
                x=agent.cross_kernel_network(x)
                finite(x,'validation CKNet')
            out=primary(net(x));finite(out,'validation logits')
            z=out.argmax(1).cpu()
            tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum())
    with torch.no_grad():
        for k,v in net.named_buffers():
            if k in bn_buffers:v.copy_(bn_buffers[k])
    net.train(previous_mode)
    if pred+ref==0:
        raise ValueError('undefined foreground Dice')
    return 2*tp/(pred+ref)


def pack_feature(feature):
    # CCD mask is shared across channels; store nonzero spatial columns losslessly.
    flat=feature.detach().cpu().reshape(feature.shape[0],-1)
    indices=torch.where(flat.ne(0).any(0))[0].to(torch.int32)
    return {'shape':tuple(feature.shape), 'indices':indices, 'values':flat[:,indices.long()].contiguous()}


def unpack_feature(item):
    c,h,w=item['shape']
    flat=torch.zeros(c,h*w)
    flat[:,item['indices'].long()]=item['values']
    return flat.reshape(c,h,w)


def replay_batch(memory,batch_size=4):
    # Preserve domain-balanced sampling and foreground preference from source.
    # Like upstream, >=1 sample/domain can exceed nominal batch size when >B domains.
    selected=[]
    per_domain=max(batch_size//len(memory),1)
    for domain in memory:
        fg=[i for i,x in enumerate(domain) if x['has_foreground']]
        nfg=min(max(per_domain//3,1),len(fg))
        ix=random.sample(fg,nfg)
        remaining=[i for i in range(len(domain)) if i not in ix]
        ix+=random.sample(remaining,per_domain-nfg)
        selected.extend(domain[i] for i in ix)
    random.shuffle(selected)
    # Upstream truncates to configured batch size.
    selected=selected[:batch_size]
    return (torch.stack([unpack_feature(s['feature']) for s in selected]).cuda(),
            torch.stack([s['target'].long() for s in selected]).cuda())


def archive_features(net, ds):
    net.eval();memory=[]
    count=min(len(ds),max(30,int(len(ds)*.05)))
    indices=random.sample(range(len(ds)),count)
    with torch.inference_mode():
        for i in indices:
            x,y=ds[i]
            out,features=net(x[None].cuda(),layer_name_for_feature_extraction='encoder')
            finite(out,'archive prediction');finite(features,'archive features')
            memory.append({'feature':pack_feature(features[0][0]),
                           'target':primary(out).argmax(1)[0].cpu().to(torch.uint8),
                           'has_foreground':bool((y==1).any())})
    return memory


def make_model(method, reference):
    if method=='focus':
        cls,gin=focus_components(Path(reference)/'FOCUS')
        net=FocusResUNet(cls)
        return net.cuda(),gin,None
    cls,agent_cls=cauaug_components(Path(reference)/'CauAug_DCMIS')
    net=resunet32(size='mid').cuda()
    agent=agent_cls()
    agent.magnitude_dataset=[];agent.phase_dataset=[];agent.std_dataset=[];agent.mean_dataset=[]
    agent.get_inputs_targets=MethodType(lambda self,data:(data[0].cuda(),data[1].cuda()),agent)
    return net,None,agent


def run():
    method=os.environ['ARM_METHOD'];seed=int(os.environ['ARM_SEED'])
    root=Path(os.environ['WORK_ROOT']);key=f'{method}_s{seed}'
    protocol=json.loads((root/'protocol.json').read_text())
    assert method in ('focus','cauaug') and seed in protocol['seeds']
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.23)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net,gin,agent=make_model(method,os.environ['REFERENCE_ROOT'])
    epochs=protocol['epochs_per_domain'][method]
    memory=[];teacher=None;selection={};stage_scores={};steps=0
    counts={'new_steps':0,'replay_steps':0,'gin_steps':0,'cknet_steps':0,'ftgen_steps':0,'kd_steps':0}
    print(json.dumps({'event':'model_ready','method':method,'seed':seed,'parameters':sum(p.numel() for p in net.parameters()),'epochs':epochs}),flush=True)
    for site in range(1,7):
        ds=dataset(os.environ['DATA_ROOT'],site,'train',method=method)
        val=dataset(os.environ['DATA_ROOT'],site,'val','selection',method=method)
        loader=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0)
        if method=='focus':
            if site>1:
                net.freeze_layers('encoder')
            opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=.0002,weight_decay=3e-5)
        else:
            opt=torch.optim.Adam(net.parameters(),lr=2e-4 if site==1 else 1e-4)
        best=-math.inf;best_state=None;best_epoch=None;best_opt=None
        for epoch in range(epochs):
            started=time.time();net.train();loss_sum=0.;n=0
            if method=='focus':
                for group in opt.param_groups:
                    group['lr']=.0002*(1-epoch/epochs)**.9
            for x,y in loader:
                x=x.cuda();y=y.cuda();opt.zero_grad(set_to_none=True)
                if method=='focus':
                    # Alternation replaces every second new-domain minibatch,
                    # matching upstream's fixed iteration budget, not doubling it.
                    replay=bool(memory) and steps%2==1
                    if replay:
                        features,yt=replay_batch(memory,batch_size=8)
                        outputs=net.feature_forward([features]);loss=dc_ce(outputs,yt)
                        counts['replay_steps']+=1
                    else:
                        if random.random()<.5:
                            with torch.no_grad():x=gin()(x)
                            counts['gin_steps']+=1
                        outputs=net(x);loss=dc_ce(outputs,y);counts['new_steps']+=1
                else:
                    with torch.no_grad():aug=agent.cross_kernel_network(x)
                    finite(aug,'CKNet');outputs=probabilities(net(aug))
                    loss=dice_bce(outputs,y);counts['cknet_steps']+=1;counts['new_steps']+=1
                    if teacher is not None:
                        # Upstream finish() freezes weights but leaves teacher BN in train mode.
                        with torch.no_grad():old=probabilities(teacher(x))
                        loss_old=agent.multi_class_cross_entropy_no_softmax(outputs,old)
                        with torch.no_grad():dom=agent.fourier_transformer_generator(x)
                        finite(dom,'FTGen');dom_out=probabilities(net(dom))
                        with torch.no_grad():dom_old=probabilities(teacher(dom))
                        loss_old=loss_old+agent.multi_class_cross_entropy_no_softmax(dom_out,dom_old)
                        loss=loss+.001*loss_old;counts['ftgen_steps']+=1;counts['kd_steps']+=1
                finite(loss,'loss');loss.backward()
                if method=='focus':
                    torch.nn.utils.clip_grad_norm_(net.parameters(),12.,error_if_nonfinite=True)
                else:
                    for p in net.parameters():
                        if p.grad is not None:finite(p.grad,'gradient')
                opt.step();loss_sum+=float(loss.detach());n+=1;steps+=1
            finite(net.state_dict(),'model state');finite(opt.state_dict(),'optimizer state')
            score=evaluate(net,val,method,agent,site,seed)
            if score>best:
                best=score;best_epoch=epoch+1
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                best_opt=copy.deepcopy(opt.state_dict())
            if method=='cauaug':
                for group in opt.param_groups:group['lr']*=.99
            progress={'event':'epoch','site':site,'epoch':epoch+1,'loss':loss_sum/n,
                      'selection_dice':score,'best_epoch':best_epoch,'seconds':time.time()-started,'counts':counts.copy()}
            print(json.dumps(progress),flush=True)
            atomic_json(root/(key+'_progress.json'),progress)
        net.load_state_dict(best_state)
        selection[str(site)]={'epoch':best_epoch,'dice':best}
        stage_scores[str(site)]=evaluate(net,dataset(os.environ['DATA_ROOT'],site,'val','holdout',method=method),method,agent,site,seed)
        if method=='focus' and site<6:
            net.freeze_layers('encoder')
            memory.append(archive_features(net,ds))
        if method=='cauaug':
            net.train()
            with torch.no_grad():
                agent.domain_knowledge_archive(DataLoader(ds,batch_size=4,shuffle=True,num_workers=0),
                    {'batch_size':4,'input_dim_c':1,'input_dim_hw':256},dataset_index=site,out_path=None)
            finite(agent.magnitude_dataset,'domain archive');finite(agent.std_dataset,'domain std')
            teacher=copy.deepcopy(net).requires_grad_(False).train()
        # Private resume evidence: one rolling checkpoint and atomic replacement.
        payload={'method':method,'seed':seed,'completed_site':site,'model':best_state,
                 'optimizer':best_opt,'selection':selection,'stage_scores':stage_scores,
                 'protocol':protocol,'counts':counts,'rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        if method=='focus':payload['replay']=memory
        else:
            payload['archive']={k:getattr(agent,k) for k in ['magnitude_dataset','phase_dataset','std_dataset','mean_dataset']}
            payload['teacher']=teacher.state_dict()
        atomic_checkpoint(root/(key+'_latest.pt'),payload)
        del best_opt,payload,ds,val,loader
        print(json.dumps({'event':'task_complete','site':site,'replay_domains':len(memory),'archive_domains':len(agent.magnitude_dataset) if agent else 0}),flush=True)
    results={str(site):evaluate(net,dataset(os.environ['DATA_ROOT'],site,'val','holdout',method=method),method,agent,site,seed) for site in range(1,7)}
    finite(results,'final scores');finite(net.state_dict(),'final model')
    atomic_json(root/(key+'.json'),{'method':method,'seed':seed,'status':'FINISHED','tasks':6,
                'epochs_per_domain':epochs,'selection':selection,'stage_scores':stage_scores,
                'results':results,'counts':counts,'protocol':protocol,
                'finite_model_optimizer':True,'test_accessed':False})
    print(json.dumps({'event':'finished','method':method,'seed':seed}),flush=True)


if __name__=='__main__':
    run()
