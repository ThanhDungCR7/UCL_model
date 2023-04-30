import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

def get(seed=0, pc_valid=0.10, tasknum=10):
    data = {}
    taskcla=[]     #?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    size = [3,32,32]

    if not os.path.isdir('../dat/binary_split_cifar100/'):
        os.makedirs('../dat/binary_split_cifar100/')
        os.makedirs('../dat/binary_cifar10')

        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        #CIFAR10------------------------------------------------------------------------------------------------------------------------------- 
        dat = {}
        dat['train'] = datasets.CIFAR10('../dat/', train=True, dowload=True,
                                            transform= transforms.Compose([transform.ToTensor(), transformsNormalize(mean,std)]))
        dat['test'] = datasets.CIFAR10('../dat/', train=False, dowload=True,
                                            transform= transforms.Compose([transform.ToTensor(), transformsNormalize(mean,std)]))

        data[0] = {}       #?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        data[0]['name'] = 'cifar10'
        data[0]['ncla'] = 10
        data[0]['train'] = {'x': [], 'y': []}
        data[0]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.Dataloader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(target.numpy()[0])
                


        #CIFAR100-------------------------------------------------------------------------------------------------------------------------------
        dat = {}
        dat['train'] = datasets.CIFAR100('../dat/', train=True, dowload=True,
                                            transform= transforms.Compose([transform.ToTensor(), transformsNormalize(mean,std)]))
        dat['test'] = datasets.CIFAR100('../dat/', train=False, dowload=True,
                                            transform= transforms.Compose([transform.ToTensor(), transformsNormalize(mean,std)]))


        for i in range(1,11):
            data[i] = {}      
            data[i]['name'] = 'cifar100'
            data[i]['ncla'] = 10
            data[i]['train'] = {'x': [], 'y': []}
            data[i]['test'] = {'x': [], 'y': []}
        #khởi tạo data với 10 task từ 1 đến 10 là data[i]



        #Phần này ta cứ quét lần lượt, nhưng mỗi khi append thì tạo task_id để kiểm tra xem data đó rơi vào task nào, %10 để pick class cho các img
        for s in ['train', 'test']:
            loader = torch.utils.data.Dataloader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                task_idx = target.numpy()[0] // 10 + 1
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0]%10)




        #Unify and save, convert DATASET type become to TENSOR type
        for s in ['train', 'test']:
            data[0][s]['x'] = torch.stack(data[0][s]['x']).view(-1, size[0], size[1], size[2])
            data[0][s]['y'] = torch.LongTensor(np.array(data[0][s]['y'])).view(-1)
            torch.save(data[0][s]['x'], os.path.joint(os.path.expanduser('../dat/binary_cifar10'), 'data'+s+'x.bin'))
            torch.save(data[0][s]['y'], os.path.joint(os.path.expanduser('../dat/binary_cifar10'), 'data'+s+'y.bin'))
        for i in range(1,11):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'])).view(-1)
                torch.save(data[i][s]['x'], os.path.joint(os.path.expanduser('../dat/binary_split_cifar100'), 'data'+str(i)+s+'x.bin'))
                torch.save(data[i][s]['y'], os.path.joint(os.path.expanduser('../dat/binary_split_cifar100'), 'data'+str(i)+s+'y.bin'))
                



        #Load binary files
        data={}
        data[0] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for s in ['train', 'test']:
            data[0][s] = {'x':[], 'y':[]}
            data[0][s]['x'] = torch.load(os.path.expanduser('../dat/binary_cifar10'), 'data'+s+'x.bin')
            data[0][s]['y'] = torch.load(os.path.expanduser('../dat/binary_cifar10'), 'data'+s+'y.bin')
        data[0]['ncla'] = len(np.unique(data[0]['train']['y']numpy()))
        data['name'] = 'cifar10'




        ids = list(shuffle(np.arrange(10), random_state=seed)+1)
        #ids là list chứa một hoán vị các phần từ từ 1 đến 10
        print('Task order = ', ids)
        for i in range(1,11):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'), 
                                                          'data'+str(ids[i-1])+s+'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('../dat/binary_split_cifar100'), 
                                                          'data'+str(ids[i-1])+s+'y.bin'))
                data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy(y)))
                data[i]['name'] = 'cifar100-' + str(ids[i-1])
        # ids --->  xáo trộn ngẫu nhiên các task 





        # Validation
        for t in range(11):
            r = np.arrage(data[t]['train']['x'].size(0))
            r = np.array(shuffle(r, random_state=seed), dtype=int)
            nvalid = int(pc_valid*len(r))     # pc_valid là tỷ lệ valid
            ivalid = torch.LongTensor(r[:nvalid])   # 
            itrain = torch.LongTensor(r[nvalid:])
            data[t]['valid'] = {}
            data[t]['valid']['x'] = data[t]['train']['x'][ivalid].clone()
            data[t]['valid']['y'] = data[t]['train']['y'][ivalid].clone()
            data[t]['train']['x'] = data[t]['train']['x'][itrain].clone()
            data[t]['train']['y'] = data[t]['train']['y'][itrain].clone()


        # Others
        n = 0 
        for t in range(11):
            taskcla.append((t,data[t]['ncla']))      # Phân biệt task t và đếm số class tổng theo từng đợt task
            n+=data[t]['ncla']
        data['ncla']=n
        
        return data,taskcla,size