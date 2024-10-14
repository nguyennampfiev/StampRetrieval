import numpy as np
import glob 
import os
from PIL import Image
from torchvision import transforms
import torch
test_transform = transforms.Compose([transforms.Resize((384, 384), Image.BICUBIC),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def retrieval_stamp(model, stamp_img, device):
	img = test_transform(stamp_img)
	img = torch.unsqueeze(img, 0)
	img = img.to(device)
	out = model(img)
	return out