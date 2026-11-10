"""Author-component TED pilot: source teacher/inference modes and standard 50 epochs.
Data split, HDF5 integer labels and explicit slice-level input adaptation retained.
"""
import os,json,copy,random,time,math
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ted_components import ted_components
from base_network import resunet32
from numerics import stable_dice_bce,stable_kd
from gradient_audit import measure
from shared import dataset,finite,atomic_json,atomic_checkpoint


def make_model():
    cls,dice,akt,agent=ted_components(Path(os.environ['REFERENCE_ROOT'])/'TED_DCMIS')
    backbone=os.environ['ARM_BACKBONE'];assert backbone in ('author','shared')
    net=cls((1,256,256),2) if backbone=='author' else resunet32(size='mid')
    return net.cuda(),dice(device='cuda:0'),akt(device='cuda:0',gugf=True),agent()


def evaluate(net,ds):
    mode=net.training;buffers={k:v.detach().clone() for k,v in net.named_buffers()}
    net.train();tp=pred=ref=0
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        for x,y in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
            out=net(x.cuda());finite(out,'evaluation');z=out.argmax(1).cpu()
            tp+=int(((z==1)&(y==1)).sum());pred+=int((z==1).sum());ref+=int((y==1).sum())
        for k,v in net.named_buffers():v.copy_(buffers[k])
    net.train(mode)
    if pred+ref==0:raise ValueError('undefined Dice')
    return 2*tp/(pred+ref)


def objective(net,teacher,x,y,seg_loss,akt,agent):
    logits=net(x);outputs=logits.softmax(1).clamp(1e-8,1-1e-8)
    variant=os.environ.get('STABILITY_VARIANT','stable_sup')
    assert variant in ('stable_sup','stable_all','legacy')
    target=F.one_hot(y,2).permute(0,3,1,2).float().contiguous()
    supervised=seg_loss(outputs,target) if variant=='legacy' else stable_dice_bce(logits,y)
    distill=outputs.new_zeros(())
    if teacher is not None:
        # Upstream freezes parameters only, retaining train-mode teacher BN.
        with torch.no_grad():old=teacher(x).softmax(1)
        image_kd=stable_kd(logits,old) if variant=='stable_all' else agent.multi_class_cross_entropy_no_softmax(outputs,old)
        noise=torch.randn_like(x)
        noise_logits=net(noise);noise_output=noise_logits.softmax(1).clamp(1e-8,1-1e-8)
        with torch.no_grad():noise_target=teacher(noise).softmax(1)
        noise_kd=stable_kd(noise_logits,noise_target) if variant=='stable_all' else agent.multi_class_cross_entropy_no_softmax(noise_output,noise_target)
        transfer=akt(outputs,old)
        distill=image_kd+noise_kd+transfer
    return supervised+float(os.environ.get('DISTILL_WEIGHT','.001'))*distill,supervised,distill


def run():
    root=Path(os.environ['WORK_ROOT']);seed=int(os.environ['ARM_SEED']);key=f'ted_s{seed}'
    protocol=json.loads((root/'protocol.json').read_text());assert seed in protocol['seeds'];epochs=protocol['epochs_per_domain']['ted']
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net,loss_f,akt,agent=make_model();teacher=None;selection={};stage={};counts={'supervised':0,'image_kd':0,'ska':0,'akt_gugf':0};history={}
    # Isolate minibatch/augmentation randomness from architecture initialization.
    stream_seed=900000+seed
    random.seed(stream_seed);np.random.seed(stream_seed);torch.manual_seed(stream_seed);torch.cuda.manual_seed_all(stream_seed)
    print(json.dumps({'event':'model_ready','seed':seed,'parameters':sum(p.numel() for p in net.parameters())}),flush=True)
    # Both LR arms start from the same preserved first-domain model.
    init_root=Path(os.environ['INIT_ROOT']);initial=torch.load(init_root/(key+'_stage1.pt'),map_location='cpu',weights_only=False)
    old_result=json.loads((init_root/(key+'.json')).read_text());assert initial['completed_site']==1
    net.load_state_dict(initial['model']);teacher=copy.deepcopy(net).requires_grad_(False).train()
    selection={'1':old_result['selection']['1']};stage={'1':old_result['stage_scores']['1']};history={'1':{'1':stage['1']}}
    atomic_checkpoint(root/(key+'_stage1.pt'),{'model':initial['model'],'completed_site':1,'selection':selection,'protocol':protocol,'reused_anchor':True})
    atomic_json(root/(key+'_stage_matrix.json'),history)
    stream_seed=910000+seed;random.seed(stream_seed);np.random.seed(stream_seed);torch.manual_seed(stream_seed);torch.cuda.manual_seed_all(stream_seed)
    print(json.dumps({'event':'shared_anchor','seed':seed,'completed_site':1,'continuation_seed':stream_seed,'cudnn_deterministic':True}),flush=True)
    for site in range(2,7):
        train=dataset(os.environ['DATA_ROOT'],site,'train',method='ted')
        val=dataset(os.environ['DATA_ROOT'],site,'val','selection',method='ted')
        later_lr=float(protocol.get('later_domain_lr',1e-4));assert 0<later_lr<=2e-4
        opt=torch.optim.Adam(net.parameters(),lr=2e-4 if site==1 else later_lr)
        loader=DataLoader(train,batch_size=4,shuffle=True,num_workers=0)
        best=-math.inf;best_state=None;best_opt=None;best_epoch=0
        for epoch in range(epochs):
            net.train();rate_used=opt.param_groups[0]['lr'];started=time.time();total=0.;sd=0.;kd=0.;grad_total=0.;n=0
            for x,y in loader:
                opt.zero_grad(set_to_none=True)
                loss,seg,dist=objective(net,teacher,x.cuda(),y.cuda(),loss_f,akt,agent)
                finite(loss,'objective')
                if teacher is not None and n==0 and epoch+1 in protocol['gradient_epochs']:
                    audit=measure(net,seg,dist,float(os.environ.get('DISTILL_WEIGHT','.001')))
                    print(json.dumps(dict(event='gradient_audit',site=site,epoch=epoch+1,**audit)),flush=True)
                loss.backward()
                for p in net.parameters():
                    if p.grad is not None:finite(p.grad,'gradient')
                grad_total+=sum(float(p.grad.detach().double().square().sum()) for p in net.parameters() if p.grad is not None)**.5
                opt.step();total+=float(loss.detach());sd+=float(seg.detach());kd+=float(dist.detach());n+=1
                counts['supervised']+=1
                if teacher is not None:
                    for k in ('image_kd','ska','akt_gugf'):counts[k]+=1
            finite(net.state_dict(),'model');finite(opt.state_dict(),'optimizer')
            score=evaluate(net,val)
            if score>best:
                best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_opt=copy.deepcopy(opt.state_dict())
            for group in opt.param_groups:group['lr']*=.99
            progress={'event':'epoch','site':site,'epoch':epoch+1,'learning_rate_used':rate_used,'loss':total/n,'segmentation_loss':sd/n,'distillation_loss':kd/n,'mean_gradient_norm':grad_total/n,'selection_dice':score,'best_epoch':best_epoch,'seconds':time.time()-started,'counts':counts.copy()}
            print(json.dumps(progress),flush=True);atomic_json(root/(key+'_progress.json'),progress)
        net.load_state_dict(best_state);selection[str(site)]={'epoch':best_epoch,'dice':best}
        stage[str(site)]=evaluate(net,dataset(os.environ['DATA_ROOT'],site,'val','holdout',method='ted'))
        history[str(site)]={str(old_site):(stage[str(site)] if old_site==site else evaluate(net,dataset(os.environ['DATA_ROOT'],old_site,'val','holdout',method='ted'))) for old_site in range(1,site+1)}
        atomic_json(root/(key+'_stage_matrix.json'),history)
        atomic_checkpoint(root/(key+f'_stage{site}.pt'),{'model':best_state,'completed_site':site,'selection':selection,'protocol':protocol})
        teacher=copy.deepcopy(net).requires_grad_(False).train();finite(teacher.state_dict(),'teacher')
        payload={'method':'ted','seed':seed,'completed_site':site,'model':best_state,'optimizer':best_opt,'teacher':teacher.state_dict(),'selection':selection,'stage_scores':stage,'counts':counts,'protocol':protocol,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        atomic_checkpoint(root/(key+'_latest.pt'),payload);del payload,best_opt,train,val,loader
        print(json.dumps({'event':'task_complete','site':site}),flush=True)
    results={str(site):evaluate(net,dataset(os.environ['DATA_ROOT'],site,'val','holdout',method='ted')) for site in range(1,7)}
    finite(results,'result');atomic_json(root/(key+'.json'),{'status':'FINISHED','method':'ted','seed':seed,'tasks':6,'epochs_per_domain':epochs,'selection':selection,'stage_scores':stage,'results':results,'counts':counts,'finite_model_optimizer':True,'test_accessed':False,'protocol':protocol})
    print(json.dumps({'event':'finished','seed':seed}),flush=True)

if __name__=='__main__':run()
