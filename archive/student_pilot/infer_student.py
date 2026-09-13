"""Offline MRI study inference from an exported YOLO student checkpoint."""
import argparse,json,os,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--validation-split',type=Path)
parser.add_argument('--limit',type=int)
args=parser.parse_args()
config_dir=args.output.parent/'yolo_config';config_dir.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(config_dir)
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np,pandas as pd,torch
import knee_mri
from student import LABELS,load_student,slice_features
torch.set_num_threads(4)
encoder,head,checkpoint=load_student(args.checkpoint)
knee_mri.DATA=args.data_dir
split='train' if args.validation_split else 'test'
if args.validation_split:
    studies=pd.read_csv(args.validation_split,dtype=str).query("split=='validation'").StudyInstanceUID.tolist()
else: studies=pd.read_csv(args.data_dir/'test.csv',dtype=str).StudyInstanceUID.tolist()
if args.limit: studies=studies[:args.limit]
series=pd.read_csv(args.data_dir/f'{split}_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
rows=[];started=time.monotonic()
with torch.inference_mode():
    for uid in studies:
        images,mask,meta=knee_mri.study_images(uid,series,split=split,size=checkpoint['image_size'])
        tensor=torch.from_numpy(images.copy()).float()[:,None].repeat(1,3,1,1)/255
        features=slice_features(encoder,tensor)[None]
        pred=head(features,torch.from_numpy(meta)[None],torch.from_numpy(mask)[None]).sigmoid().numpy()[0]
        assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
        rows.append(dict(StudyInstanceUID=uid,**dict(zip(LABELS,pred.tolist()))))
frame=pd.DataFrame(rows,columns=['StudyInstanceUID']+LABELS)
assert frame.StudyInstanceUID.tolist()==studies
if not args.validation_split and not args.limit:
    sample=pd.read_csv(args.data_dir/'sample_submission.csv')
    assert list(frame.columns)==list(sample.columns) and set(frame.StudyInstanceUID)==set(sample.StudyInstanceUID)
frame.to_csv(args.output,index=False)
elapsed=time.monotonic()-started
print(json.dumps({'studies':len(studies),'seconds':elapsed,'seconds_per_study':elapsed/max(1,len(studies)),
                  'device':'cpu','output':str(args.output)}))
