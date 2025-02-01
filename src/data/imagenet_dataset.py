import torch
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop

import os
path = os.path.join(os.path.dirname(__file__), '../..')
imagenet_path = os.path.join(path, 'imagenet10')

def imagenet_dataset(split):
  transforms = Compose([
    CenterCrop(size=(256, 256)),
    ToTensor(),
    Normalize(mean=torch.tensor([0.4815, 0.4578, 0.4082]), std=torch.tensor([0.2686, 0.2613, 0.2758]))]
  )
  return ImageNet(imagenet_path, split=split, transform=transforms)
