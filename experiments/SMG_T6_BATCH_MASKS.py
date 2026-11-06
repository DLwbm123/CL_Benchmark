import torch

def summarize(y):
    if y.ndim!=3 or not y.shape[0] or not torch.isfinite(y).all(): raise ValueError('invalid masks')
    if ((y!=0)&(y!=1)).any():raise ValueError('nonbinary masks')
    foreground=(y==1)
    return {'slices':len(y),'empty_slices':int((~foreground.flatten(1).any(1)).sum()),'foreground_fraction':float(foreground.double().mean())}
if __name__=='__main__':
 import os,json
 from pathlib import Path
 from torch.utils.data import DataLoader,Subset
 from datasets.seq_prostate import ProstateCL_Dataset
 root=os.environ.get('CL_DATA_ROOT')
 if not root or not (Path(root)/'Domain_Prostate'/'I2CVB.h5').is_file():raise RuntimeError('training data root is unavailable')
 ds=Subset(ProstateCL_Dataset('6','train',256),list(range(24)))
 rows=[summarize(y) for _,y in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)]
 if len(rows)!=3 or any(x['slices']!=8 for x in rows):raise RuntimeError('incomplete batches')
 out={'batches':rows,'post_hoc':True,'split':'site6 train first 24','validation_used':False,'test_used':False}
 Path(os.environ['WORK_ROOT'],'result.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 print('three batch aggregates completed',flush=True)
