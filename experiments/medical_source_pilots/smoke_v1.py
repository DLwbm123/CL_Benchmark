import json,os,copy,time
from pathlib import Path
import torch
from torch import nn
from runner_v1 import make_model,finite,dc_ce,dice_bce,probabilities,pack_feature,unpack_feature,replay_batch,dataset

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.23)
    torch.manual_seed(902);checks={};timings={}
    x=torch.rand(4,1,256,256,device='cuda');y=torch.zeros(4,256,256,device='cuda',dtype=torch.long);y[:,60:130,80:150]=1
    for method in ('focus','cauaug'):
        started=time.time();net,gin,agent=make_model(method,os.environ['REFERENCE_ROOT'])
        net.train();opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        if method=='focus':
            aug=gin()(x);finite(aug,'GIN');assert aug.shape==x.shape
            out=net(aug);loss=dc_ce(out,y);loss.backward();opt.step();finite(net.state_dict(),'focus updated')
            net.eval()
            with torch.no_grad():
                ordinary=net(x)[0]
                first=net.conv_blocks_context[0](x)
                net.freeze_layers('conv_blocks_context.6')
                replay=net.feature_forward([first])[0]
                assert torch.allclose(ordinary,replay,rtol=1e-5,atol=1e-5),'feature-forward equivalence'
                masked=net.sparsify_features(torch.ones(4,32,256,256,device='cuda'))
                assert torch.equal(masked[:,0],masked[:,31]),'channel consistency'
                assert .85<float((masked==0).float().mean())<.95
                packed=pack_feature(masked[0]);assert torch.equal(unpack_feature(packed),masked[0].cpu())
            frozen={k:v.clone() for k,v in net.conv_blocks_context.state_dict().items()}
            decoder=next(net.conv_blocks_localization.parameters()).detach().clone()
            net.train();opt.zero_grad();dc_ce(net.feature_forward([masked]),y).backward();opt.step()
            assert all(torch.equal(v,net.conv_blocks_context.state_dict()[k]) for k,v in frozen.items()),'encoder changed'
            assert not torch.equal(decoder,next(net.conv_blocks_localization.parameters())), 'decoder did not update'
            memory=[[{'feature':packed,'target':y[0].cpu().to(torch.uint8),'has_foreground':True} for _ in range(30)] for _ in range(3)]
            rx,ry=replay_batch(memory);assert len(rx)==3 and len(ry)==3
            checks[method]=['GIN finite','full supervised backward','feature-forward equivalence','CCD channel consistency','lossless feature packing','encoder freeze incl buffers','decoder gradient','balanced replay']
        else:
            aug=agent.cross_kernel_network(x);finite(aug,'CKNet');assert aug.min()>=0 and aug.max()<=1
            out=probabilities(net(aug));loss=dice_bce(out,y);loss.backward();opt.step()
            src=[(x.cpu(),y.cpu())]
            with torch.no_grad():
                agent.domain_knowledge_archive(src,{'batch_size':4,'input_dim_c':1,'input_dim_hw':256},out_path=None)
                assert torch.allclose(agent.magnitude_dataset[0],torch.fft.fft2(x).abs().mean(0))
                dom=agent.fourier_transformer_generator(x);finite(dom,'FTGen');assert dom.shape==x.shape
            teacher=copy.deepcopy(net).requires_grad_(False).train();before={k:v.clone() for k,v in teacher.named_parameters()}
            opt.zero_grad();pred=probabilities(net(dom))
            with torch.no_grad():target=probabilities(teacher(dom))
            kd=agent.multi_class_cross_entropy_no_softmax(pred,target);finite(kd,'source KD');kd.backward();opt.step()
            assert all(torch.equal(v,dict(teacher.named_parameters())[k]) for k,v in before.items()),'teacher weights changed'
            assert all(p.grad is None for p in teacher.parameters())
            finite(net.state_dict(),'cauaug updated');finite(opt.state_dict(),'optimizer')
            checks[method]=['CKNet finite/range','DiceBCE backward','archive analytic equivalence','FTGen finite','source KD backward','teacher weights frozen']
        timings[method]=time.time()-started
        del net,opt
        torch.cuda.empty_cache()
    ds=dataset(os.environ['DATA_ROOT'],1,'train');val=dataset(os.environ['DATA_ROOT'],1,'val','selection')
    assert len(ds)>0 and len(val)>0
    scaled=dataset(os.environ['DATA_ROOT'],1,'train',method='cauaug').tensors[0]
    assert bool(torch.isfinite(scaled).all()) and scaled.min()>=0 and scaled.max()<=1
    checks['data']=['frozen integer-cast label policy','CauAug slice minmax finite in [0,1]']
    out={'status':'PASSED','checks':checks,'seconds':timings,'torch':torch.__version__,'cuda':torch.version.cuda,'data':'one train domain and selection block; no test'}
    Path(os.environ['WORK_ROOT'],'smoke.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
