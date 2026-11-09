import os,json,copy,torch
from pathlib import Path
from runner import make_model,dc_ce,finite
root=Path(os.environ['WORK_ROOT']);torch.set_num_threads(2);torch.manual_seed(41)
net,gin,_=make_model('focus',os.environ['REFERENCE_ROOT']);x=torch.randn(4,1,256,256,device='cuda');y=torch.zeros(4,256,256,dtype=torch.long,device='cuda');y[:,70:150,70:150]=1
# Bypass consumes precisely the same RNG as CCD but preserves tensor values.
state=torch.cuda.get_rng_state();original=net.sparsify_features
z=original(x,during_normal_training=True);after=torch.cuda.get_rng_state();torch.cuda.set_rng_state(state)
def bypass(a,**kw):original(a,**kw);return a
assert torch.equal(bypass(x,during_normal_training=True),x) and torch.equal(after,torch.cuda.get_rng_state())
assert not torch.equal(z,x)
initial=copy.deepcopy(net.state_dict())
for variant in ['full_sgd','no_ccd','no_gin','full_adam']:
 net.load_state_dict(initial);net.train();net.sparsify_features=bypass if variant=='no_ccd' else original
 opt=torch.optim.Adam(net.parameters(),lr=.0002,weight_decay=3e-5) if variant=='full_adam' else torch.optim.SGD(net.parameters(),lr=.01,momentum=.99,nesterov=True,weight_decay=3e-5)
 with torch.no_grad():aug=gin()(x)
 opt.zero_grad(set_to_none=True);loss=dc_ce(net(x if variant=='no_gin' else aug),y);finite(loss,'smoke loss');loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),12.,error_if_nonfinite=True);opt.step();finite(net.state_dict(),'smoke model');finite(opt.state_dict(),'smoke optimizer')
(root/'smoke.json').write_text(json.dumps({'status':'PASSED','variants':4,'ccd_bypass_preserves_rng_and_identity':True,'finite_backward_update':True}))
print('PASSED',flush=True)
