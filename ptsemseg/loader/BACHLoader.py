import os
import torch
import numpy as np
import scipy.misc as m
import operator 
from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *


SEG_LABELS_LIST = [
    {"id": 3, "name": "void",       "rgb_values": [0,   0,    0]},
    {"id": 0,  "name": "benign",   "rgb_values": [250, 0,    0]},
    {"id": 1,  "name": "in situ",      "rgb_values": [0,   150,  0]},
    {"id": 2,  "name": "invasive",       "rgb_values": [0, 0,  70]}]


class BACHLoader(data.Dataset):
    """BACH

    https://iciar2018-challenge.grand-challenge.org/

    Data is derived from BACH, and can be downloaded from here:
    https://iciar2018-challenge.grand-challenge.org/dataset/
    """
    colors = [#[  0,   0,   0],
              [250,   0,    0],
              [  0, 150,    0],
              [  0,   0,   70]]
     

    label_colours = dict(zip(range(3), colors))

    def __init__(self, root, split="train", is_transform=False, 
                 img_size=(1024, 1024), augmentations=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations 
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = None
        self.n_classes = 4
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')
        
        self.void_classes = [0]
        self.valid_classes = [250,150,70]
        self.class_names = ['benign', 'in situ', 'invasive']

        self.ignore_index = 3
        self.class_map = dict(zip(self.valid_classes, range(3))) 

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """ 
        img_path = os.path.join(self.annotations_base,
                                'train/' + 'A0'+str(index+1)+'_input.png')

        print('Reading images from:',img_path,index)
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2], 
                                'A0'+str(index+1)+'.png')

        print('Reading label from:',lbl_path)
        img = m.imread(img_path)
        #print('size of img:',img.shape)
        img = np.array(img, dtype=np.int8)

        lbl = m.imread(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        unique, counts = np.unique(lbl, return_counts=True)
        x=dict(zip(unique,counts))
        y=sorted(x.items(), key=operator.itemgetter(1))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        
        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)
        classes = np.unique(lbl)
        #print('classes:',classes)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl!=3]) < self.n_classes):
            print('after det', classes,  np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, target):
        target_labels = target[..., 0]
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']
        print('target labels shape',target_labels.shape)
        unique, counts = np.unique(target_labels, return_counts=True)
	x=dict(zip(unique,counts))
	y=sorted(x.items(), key=operator.itemgetter(1))
	print('Unique counts of input labels:',y)


        return target_labels  

if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048),
                             RandomRotate(10),
                             RandomHorizontallyFlip()])

    local_path = '/home/deepita/pytorch-semseg-master/ptsemseg/dataset/'
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs,2)
        for j in range(bs):      
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show(block=True)
        a = raw_input()
        if a == 'ex':
            break
        else:
            plt.close()
