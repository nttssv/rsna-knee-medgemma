"""Paired frozen-YOLO student pilot, using the teacher's fixed group split."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--teacher-run',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--prepare-only',action='store_true')
parser.add_argument('--epochs',type=int,default=20)
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
config_dir=args.output_dir/'yolo_config';config_dir.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(config_dir)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from ultralytics import settings
import ultralytics
settings.update({'sync':False})
import knee_mri
from student import LABELS,StudyHead,make_encoder,slice_features,load_student

torch.set_num_threads(4)
SEED=42
def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
seed()
knee_mri.DATA=args.data_dir
split=pd.read_csv(args.teacher_run/'development_split.csv',dtype=str)
assert split.StudyInstanceUID.is_unique
truth=pd.read_csv(args.data_dir/'train.csv').set_index('StudyInstanceUID').loc[split.StudyInstanceUID]
series=pd.read_csv(args.data_dir/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
train_mask=split.split.eq('train').to_numpy();val_mask=split.split.eq('validation').to_numpy()
assert set(split.loc[train_mask,'group']).isdisjoint(split.loc[val_mask,'group'])
cache_path=args.output_dir/'features.pt'
weights=args.output_dir/'yolo26n-cls.pt'
if not cache_path.exists():
    # Download through the official Ultralytics loader into this persistent directory.
    previous=os.getcwd();os.chdir(args.output_dir)
    try: encoder=make_encoder(weights='yolo26n-cls.pt')
    finally: os.chdir(previous)
    features=[];masks=[];metadata=[]
    started=time.monotonic()
    with torch.inference_mode():
        for i,uid in enumerate(split.StudyInstanceUID):
            images,mask,meta=knee_mri.study_images(uid,series,size=224)
            tensor=torch.from_numpy(images.copy()).float()[:,None].repeat(1,3,1,1)/255
            features.append(slice_features(encoder,tensor))
            masks.append(torch.from_numpy(mask));metadata.append(torch.from_numpy(meta))
            if (i+1)%10==0 or i+1==len(split): print(f'YOLO features: {i+1}/{len(split)} studies',flush=True)
    cache=dict(features=torch.stack(features),mask=torch.stack(masks),metadata=torch.stack(metadata),
               study_ids=split.StudyInstanceUID.tolist(),architecture=encoder.yaml,
               encoder_state=encoder.state_dict(),feature_seconds=time.monotonic()-started,
               upstream_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
               preprocessing_sha256=hashlib.sha256(Path(knee_mri.__file__).read_bytes()).hexdigest())
    torch.save(cache,cache_path)
else:
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
assert cache['study_ids']==split.StudyInstanceUID.tolist()
assert cache['preprocessing_sha256']==hashlib.sha256(Path(knee_mri.__file__).read_bytes()).hexdigest()
if args.prepare_only:
    print('Student features ready; no student training requested.',flush=True)
    raise SystemExit(0)

targets=pd.read_csv(args.teacher_run/'teacher_targets_train_in_sample.csv').set_index('StudyInstanceUID')
assert set(targets.index)==set(split.loc[train_mask,'StudyInstanceUID'])
assert set(targets.index).isdisjoint(set(split.loc[val_mask,'StudyInstanceUID']))
y=torch.tensor(truth[LABELS].to_numpy(dtype=np.float32).copy())
known=torch.isfinite(y);y=torch.nan_to_num(y)
teacher=torch.full_like(y,0.5)
teacher[train_mask]=torch.tensor(targets.loc[split.loc[train_mask,'StudyInstanceUID'],LABELS].to_numpy(dtype=np.float32).copy())
assert torch.isfinite(teacher).all() and ((teacher>=0)&(teacher<=1)).all()
x,meta,mask=cache['features'],cache['metadata'],cache['mask']
feature_dim=x.shape[-1]
seed();initial=StudyHead(feature_dim).state_dict()
results=[]
for name,kd_weight in [('labels_only',0.0),('distilled',0.3)]:
    seed();head=StudyHead(feature_dim);head.load_state_dict(copy.deepcopy(initial))
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-3)
    indices=np.flatnonzero(train_mask);rng=np.random.default_rng(SEED)
    history=[];started=time.monotonic()
    for epoch in range(args.epochs):
        head.train();losses=[]
        for chunk in np.array_split(rng.permutation(indices),max(1,int(np.ceil(len(indices)/16)))):
            logits=head(x[chunk],meta[chunk],mask[chunk])
            raw=F.binary_cross_entropy_with_logits(logits,y[chunk],reduction='none')
            supervised=(raw*known[chunk]).sum()/known[chunk].sum().clamp_min(1)
            temperature=2.0
            soft=torch.sigmoid(torch.logit(teacher[chunk].clamp(1e-5,1-1e-5))/temperature)
            kd=F.binary_cross_entropy_with_logits(logits/temperature,soft)*temperature**2
            loss=(1-kd_weight)*supervised+kd_weight*kd
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        pd.DataFrame(history).to_csv(args.output_dir/f'{name}_history.csv',index=False)
        with (args.output_dir/f'{name}_metrics.jsonl').open('a') as log:
            log.write(json.dumps(dict(step=epoch+1,values={'student/train_loss':history[-1]['loss']}))+'\n')
        print(f'{name}: epoch {epoch+1}/{args.epochs}, loss={history[-1]["loss"]:.4f}',flush=True)
    head.eval()
    with torch.inference_mode(): scores=head(x[val_mask],meta[val_mask],mask[val_mask]).sigmoid().numpy()
    actual=truth[LABELS].to_numpy(dtype=float)[val_mask]
    for j,label in enumerate(LABELS):
        available=np.isfinite(actual[:,j])
        auc=roc_auc_score(actual[available,j],scores[available,j]) if len(np.unique(actual[available,j]))==2 else None
        results.append(dict(model=name,label=label,n=int(available.sum()),auc=auc))
    checkpoint=dict(labels=LABELS,architecture=cache['architecture'],encoder_state=cache['encoder_state'],
                    feature_dim=feature_dim,head_state=head.state_dict(),image_size=224,slices_per_plane=2,
                    upstream_weights_sha256=cache['upstream_weights_sha256'],
                    config=dict(encoder='yolo26n-cls',frozen_encoder=True,epochs=args.epochs,seed=SEED,
                                kd_weight=kd_weight,temperature=2.0,training_studies=int(train_mask.sum()),
                                validation_studies=int(val_mask.sum()),ultralytics=str(ultralytics.__version__),
                                teacher_targets='in_sample_training_studies_only',training_seconds=time.monotonic()-started))
    path=args.output_dir/f'student_{name}.pt';torch.save(checkpoint,path)
    auc_values=[r['auc'] for r in results if r['model']==name and r['auc'] is not None]
    with (args.output_dir/f'{name}_metrics.jsonl').open('a') as log:
        log.write(json.dumps(dict(step=args.epochs,values={'validation/mean_auc':float(np.mean(auc_values))}))+'\n')
    # Round trip verifies the exported architecture and tensors are sufficient offline.
    loaded_encoder,loaded_head,_=load_student(path)
    with torch.inference_mode(): restored=loaded_head(x[val_mask],meta[val_mask],mask[val_mask]).sigmoid().numpy()
    np.testing.assert_allclose(restored,scores,rtol=1e-6,atol=1e-6)
pd.DataFrame(results).to_csv(args.output_dir/'validation_metrics.csv',index=False)
means=pd.DataFrame(results).groupby('model').auc.mean().to_dict()
(args.output_dir/'student_status.json').write_text(json.dumps(dict(status='complete',models=means,
    training_studies=int(train_mask.sum()),validation_studies=int(val_mask.sum()),frozen_encoder=True),indent=2))
print(json.dumps(means),flush=True)
