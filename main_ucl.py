import sys, os, time
import numpy as np
import torch
import torch.nn as nn

import utils
from utils import *
from arguments import get_args

directory = "./result_data/csvdata/"
if not os.path.exists(directory):
    os.makedirs(directory)


tstart = time.time()

log_name = '{}_{}_{}_{}_alpha_{}_beta_{:.5f}_ratio_{:.4f}_lr_{}_lr_rho_{}_unitN_{}_batch_{}_epoch_{}'.format(
        args.date, args.experiment, args.approach, args.seed, args.alpha, args.beta, args.ratio, 
        args.lr, args.lr_rho, args.unitN, args.batch_size, args.nepochs)

#argument
print('=' * 100)
print('Arguments = ')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args,arg))
print('=' * 100)


#################################################################################################################
# Split 
split = False 
notMNIST = False
split_experiment = [      
    'split_mnist', 
    'split_notmnist', 
    'split_cifar10',
    'split_cifar100',
    'split_cifar100_20',
    'split_cifar10_100',
    'split_pmnist',
    'split_row_pmnist', 
    'split_CUB200',
    'split_tiny_imagenet',
    'split_mini_imagenet', 
    'omniglot',
    'mixture'
]

conv_experiment = [
    'split_cifar10',
    'split_cifar100',
    'split_cifar100_20',
    'split_cifar10_100',
    'split_CUB200',
    'split_tiny_imagenet',
    'split_mini_imagenet', 
    'omniglot',
    'mixture'
]

if args.experiment in split_experiment:   #exp trong tập này thì tức là exp -> split
    split = True
if args.experiment == 'split_notmnist':   #exp = 'split_notmnist' thì tức chuyển notMNIST = True
    notMNIST = True
if args.experiment in conv_experiment:   #exp trong tập này thì tức là exp -> conv
    args.conv = True
    log_name = log_name + '_conv'
if args.output == '':                   #output rỗng thì tạo output = './result_data/' + log_name + '.txt'
    args.output = './result_data/' + log_name + '.txt'


#FIX CỨNG args (Phần này làm tổng quát thì input cả exp và approach)
args.experiment = 'split_cifar10_100'
args.approach = 'ucl'  


# SEED
np.random.seed(args.seed)  #--> args.seed khởi tạo ban đầu như exp và approaches, mặc định bên file arg thì seed = 0
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA is unavailable]'); sys.exit()



# Args --> Exp
if args.experiment == 'mnist2':
    from dataloaders import mnist2 as dataloader
elif args.experiment == 'pmnist' or args.experiment == 'split_pmnist':
    from dataloaders import pmnist as dataloader
elif args.experiment == 'row_pmnist' or args.experiment == 'split_row_pmnist':
    from dataloaders import row_pmnist as dataloader
elif args.experiment == 'split_mnist':
    from dataloaders import split_mnist as dataloader
elif args.experiment == 'split_notmnist':
    from dataloaders import split_notmnist as dataloader
elif args.experiment == 'split_cifar10':
    from dataloaders import split_cifar10 as dataloader
elif args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader
elif args.experiment == 'split_cifar100_20':
    from dataloaders import split_cifar100_20 as dataloader
elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader
elif args.experiment == 'split_CUB200':
    from dataloaders import split_CUB200 as dataloader
elif args.experiment == 'split_tiny_imagenet':
    from dataloaders import split_tiny_imagenet as dataloader
elif args.experiment == 'split_mini_imagenet':
    from dataloaders import split_mini_imagenet as dataloader
elif args.experiment == 'omniglot':
    from dataloaders import split_omniglot as dataloader
elif args.experiment == 'mixture':
    from dataloaders import mixture as dataloader

#------------------->>>>>>>>> Đã chọn exp = split_cifar10_100 nên sẽ RUN -> "from dataloaders import split_cifar10_100 as dataloader"



# Args --> Approach
if args.approach == 'random':
    from approaches import random as approach
elif args.approach == 'ucl':
    from approaches import ucl as approach
elif args.approach == 'ucl_ablation':
    from approaches import ucl_ablation as approach
elif args.approach == 'baye_hat':
    from core import baye_hat as approach
elif args.approach == 'baye_fisher':
    from core import baye_fisher as approach
elif args.approach == 'sgd':
    from approaches import sgd as approach
elif args.approach == 'sgd-restart':
    from approaches import sgd_restart as approach
elif args.approach == 'sgd-frozen':
    from approaches import sgd_frozen as approach
elif args.approach == 'sgd_with_log':
    from approaches import sgd_with_log as approach
elif args.approach == 'sgd_L2_with_log':
    from approaches import sgd_L2_with_log as approach
elif args.approach == 'lwf':
    from approaches import lwf as approach
elif args.approach == 'lwf_with_log':
    from approaches import lwf_with_log as approach
elif args.approach == 'lfl':
    from approaches import lfl as approach
elif args.approach == 'ewc':
    from approaches import ewc as approach
elif args.approach == 'si':
    from approaches import si as approach
elif args.approach == 'rwalk':
    from approaches import rwalk as approach
elif args.approach == 'mas':
    from approaches import mas as approach
elif args.approach == 'imm-mean':
    from approaches import imm_mean as approach
elif args.approach == 'imm-mode':
    from approaches import imm_mode as approach
elif args.approach == 'progressive':
    from approaches import progressive as approach
elif args.approach == 'pathnet':
    from approaches import pathnet as approach
elif args.approach == 'hat-test':
    from approaches import hat_test as approach
elif args.approach == 'hat':
    from approaches import hat as approach
elif args.approach == 'joint':
    from approaches import joint as approach


#------------------->>>>>>>>> Đã chọn approach = ucl nên sẽ RUN -> "from approaches import ucl as approach"



# Args --> Network
if args.experiment == 'split_cifar100' or args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar10' or args.experiment == 'split_cifar100_20':
    if args.approach == 'hat':
        from networks import conv_net_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import conv_net_ucl as network
    else:
        from networks import conv_net as network

elif args.experiment == 'mixture' or args.experiment == 'split_mini_imagenet' or args.experiment == 'split_tiny_imagenet' or args.experiment == 'split_CUB200':
    if args.approach == 'hat':
        from networks import alexnet_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import alexnet_ucl as network
    else:
        from networks import alexnet as network

elif args.experiment == 'omniglot':
    if args.approach == 'hat':
        from networks import conv_net_omniglot_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import conv_net_omniglot_ucl as network
    else:
        from networks import conv_net_omniglot as network
else:
    if args.approach == 'hat':
        from networks import mlp_hat as network
    elif args.approach == 'ucl' or args.approach == 'ucl_ablation':
        from networks import mlp_ucl as network
    else:
        from networks import mlp as network
    

    # approach = ucl, exp = cifar10_100 ======> network = conv_net_ucl