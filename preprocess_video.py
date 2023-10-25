import sys
#import torch as th

import torchvision.models as models
from utilities import resnext

#from torch import nn

from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import numpy as np

import os
import random
import math

from utilities.video_loader import VideoLoader
from utilities.preprocessing import Preprocessing

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

model_name = "resnet50"
model_type = "2d"

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])
    
def get_model(model_type, model, resnext101_model_path):
    assert model_type in ['2d', '3d']
    if model_type == '2d':
        if model == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Identity()
            #model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        elif model == 'resnet101':
            model = models.resnet101(pretrained=True)
            model.fc = nn.Identity()
        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Identity()
        elif model == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Identity()
        elif model == 'densenet169':
            model = models.densenet169(pretrained=True)
            model.classifier = nn.Identity()
        elif model == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Identity()
        elif model == 'vgg19':
            model = models.vgg19(pretrained=True)
            model.classifier[6] = nn.Identity()
        elif model == 'alexnet':
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Identity()
        elif model == 'inception_v3':
            model = models.inception_v3(pretrained=True, aux_logits=False)
            model.fc = nn.Identity()
        elif model == 'resnext50':
            model = models.resnext50_32x4d(pretrained=True)
            model.fc = nn.Identity()
        elif model == 'resnext101':
            model = models.resnext101_32x8d(pretrained=True)
            model.fc = nn.Identity()
        elif model == 'shufflenet':
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.fc = nn.Identity()
        elif model == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Identity()
        elif model == 'mnasnet':
            model = models.mnasnet1_0(pretrained=True)
            model.classifier[1] = nn.Identity()
        elif model == 'efficientnet-b0' :
            model = EfficientNet.from_pretrained('efficientnet-b0')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b1' :
            model = EfficientNet.from_pretrained('efficientnet-b1')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b2' :
            model = EfficientNet.from_pretrained('efficientnet-b2')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b3' :
            model = EfficientNet.from_pretrained('efficientnet-b3')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b4' :
            model = EfficientNet.from_pretrained('efficientnet-b4')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b5' :
            model = EfficientNet.from_pretrained('efficientnet-b5')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b6' :
            model = EfficientNet.from_pretrained('efficientnet-b6')
            model._fc = nn.Identity()
        elif model == 'efficientnet-b7' :
            model = EfficientNet.from_pretrained('efficientnet-b7')
            model._fc = nn.Identity()
        model = model.cuda()
        
    else:
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = torch.load(resnext101_model_path)
        model.load_state_dict(model_data)

    model.eval()
    print('loaded')
    return model

def main():
    flist = []
    directory_video = "dataset/vevo/"
    for path, subdirs, files in os.walk( directory_video ):
        for fname in files:
            filepath = os.path.join(path, fname)
            flist.append(filepath)

    l2_normalize = 1
    half_precision = 1
    batch_size=64
    if model_name == "inception_v3":
        dataset = VideoLoader(
            fileList = flist,
            framerate=1 if model_type == '2d' else 24,
            size=299 if model_type == '2d' else 149,
            centercrop=(model_type == '3d'),
        )
    else:
        dataset = VideoLoader(
            fileList = flist,
            framerate=1 if model_type == '2d' else 24,
            size=224 if model_type == '2d' else 112,
            centercrop=(model_type == '3d'),
        )
        
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=None
    )

    preprocess = Preprocessing(model_type)
    model = get_model(model_type, model_name, "resnext101.pth")

    device = torch.device("cuda:3")
    sample = dataset[0]
    print(sample["video"].shape)

    o = model(sample["video"][0:1].cuda(device))
    fsize = o.shape[1]

    with torch.no_grad():
        for k, data in enumerate(loader):
            input_file = data['input'][0]
            output_file = data['output'][0]
            output_filename = output_file.split("/")[-1]

            output_file = os.path.join("dataset/vevo_vis/all_v2", model_type, model_name, output_filename)
            os.makedirs(os.path.join("dataset/vevo_vis/all_v2", model_type, model_name), exist_ok=True)        

            if len(data['video'].shape) > 3:
                video = data['video'].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    features = torch.cuda.FloatTensor(n_chunk, fsize).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(batch_size)))
                    for i in range(n_iter):
                        min_ind = i * batch_size
                        max_ind = (i + 1) * batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        batch_features = model(video_batch)
                        if l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if half_precision:
                        features = features.astype('float16')
                    np.save(output_file, features)
            else:
                print('Video {} already processed.'.format(input_file))
    

if __name__ == "__main__":
    main()
