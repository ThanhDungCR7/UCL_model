import sys, time, os
import numpy as np 
import random
import torch
from copy import deepcopy
import utils
from utils import *
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import math

sys.path.append('..')
from arguments import get_args
args = get_args()

from bayes_layer import BayesianLinear, BayesianConv2D, _calculate_fan_in_and_fan_out


class Appr(object):

    def __init__(self, model, nepochs=100, sbatch=256, lr=0.001,
                 lr_min=2e-6, lr_factor=3, lr_patience=5, clipgrad=100, args=None, log_name=None, split=False):
        return
    