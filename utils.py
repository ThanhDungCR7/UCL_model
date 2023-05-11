import os,sys
import numpy as np
import random
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from torch._six import inf
import pandas as pd
from PIL import Image
from sklearn.feature_extraction import image
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import *
from arguments import get_args
args = get_args()

resnet_model = models.resnet18(pretrained=True).cuda()
feature_extractor = nn.Sequential(*list(resnet_model.children())[:-4])

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################
def crop(x, patch_size, mode = 'train'):
    cropped_image = []
    arr_len = len(x)
    if mode == 'train':
        for idx in range(arr_len):
            
            patch = image.extract_patches_2d(image = x[idx].data.cpu().numpy(),
                                            patch_size = (patch_size, patch_size), max_patches = 1)[0]
            
            # Random horizontal flipping
            if random.random() > 0.5:
                patch = np.fliplr(patch)
            # Random vertical flipping
            if random.random() > 0.5:
                patch = np.flipud(patch)
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)
    elif mode == 'valid' or mode == 'test':
        for idx in range(arr_len):
            patch = x[idx].data.cpu().numpy()
            H,W,C = patch.shape
            patch = patch[H//2-patch_size//2:H//2+patch_size//2, W//2-patch_size//2:W//2+patch_size//2,:]
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)
        
    image_tensor=torch.stack(cropped_image).view(-1,3,patch_size,patch_size).cuda()
    return image_tensor


def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return



###############################################################################################################################################################
class logger(object):
    def __init__(self, file_name='pmnist2', resume=False, path='./result_data/csvdata/', data_format='csv'):
        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))
