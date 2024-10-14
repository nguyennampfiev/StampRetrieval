import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# Set deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch.nn as nn
import matplotlib.pyplot as plt


from models.build import build_models, freeze_backbone
from defaults import _C
from util import SetSeed

def get_config():
    config = _C.clone()
    cfg_file = os.path.join('configs/swin-cham.yaml')
    config.merge_from_file(cfg_file)

    config.cuda_visible = '0'

    config.model.pretrained = 'ckpt/Swin Base.pth'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
    os.environ['OMP_NUM_THREADS'] = '1'


    config.freeze()
    SetSeed(config)
    return config

def build_retrieval_model(config, num_classes):
    """
    Builds and returns the model based on the provided configuration.
    """
    model = build_models(config, num_classes)
    #print(config.device)
    #model.to(config.device)
    freeze_backbone(model, config.train.freeze_backbone)
    model = load_checkpoint(config, model)
    model.eval()
    model.head=nn.Identity()
    model.head_drop=nn.Identity()
    return model

def load_checkpoint(config, model):
	checkpoint = torch.load('ckpt/checkpoint.bin', map_location='cpu')
	state_dicts = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
	state_dicts = {k.replace('_orig_mod.', ''): v for k, v in state_dicts.items()}
	model.load_state_dict(state_dicts, strict=True)
	return model
