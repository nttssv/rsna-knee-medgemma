"""Small YOLO26 slice encoder with a study-level multilabel attention head.

Research pilot: the ImageNet encoder is frozen; only the study head is trained.
Sources: https://docs.ultralytics.com/tasks/classify and /reference/nn/tasks
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F

LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
        'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

def make_encoder(config=None,weights=None):
    if weights is not None:
        from ultralytics import YOLO
        encoder=YOLO(str(weights),task='classify').model
    else:
        from ultralytics.nn.tasks import ClassificationModel
        encoder=ClassificationModel(cfg=copy.deepcopy(config),verbose=False)
    encoder.model=encoder.model[:-1]  # Remove the ImageNet softmax classification head.
    encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad=False
    return encoder

def slice_features(encoder,images):
    return F.adaptive_avg_pool2d(encoder(images),(1,1)).flatten(1)

class StudyHead(nn.Module):
    def __init__(self,feature_dim,hidden=128):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(feature_dim+6,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.attention=nn.Linear(hidden,len(LABELS))
        self.weight=nn.Parameter(torch.empty(len(LABELS),hidden))
        self.bias=nn.Parameter(torch.zeros(len(LABELS)))
        nn.init.normal_(self.weight,std=0.02)

    def forward(self,features,metadata,mask):
        assert mask.any(1).all(), 'Each study needs at least one usable slice.'
        hidden=self.project(torch.cat([features,metadata],dim=-1))
        weights=self.attention(hidden).masked_fill(~mask[:,:,None],float('-inf')).softmax(dim=1)
        pooled=torch.einsum('bsl,bsh->blh',weights,hidden)
        return (pooled*self.weight[None]).sum(-1)+self.bias

def load_student(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    assert checkpoint['labels']==LABELS
    encoder=make_encoder(config=checkpoint['architecture'])
    encoder.load_state_dict(checkpoint['encoder_state'],strict=True)
    head=StudyHead(checkpoint['feature_dim'])
    head.load_state_dict(checkpoint['head_state'],strict=True)
    return encoder.to(device).eval(),head.to(device).eval(),checkpoint
