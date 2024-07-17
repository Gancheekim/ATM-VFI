"""
code borrowed from RIFE: https://github.com/megvii-research/ECCV2022-RIFE/blob/main/dataset.py
"""
import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

cv2.setNumThreads(0)
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, path, scale_factor=1, train_crop=None):
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.scale_factor = scale_factor
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

        self.train_crop = train_crop
        if self.train_crop is None:
            if self.scale_factor == 1:
                self.train_crop = 256
            elif self.scale_factor == 2:
                self.train_crop = 384
            else:
                self.train_crop = 448

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])

        if self.scale_factor > 1:
            img0 = cv2.resize(img0, (int(self.w * self.scale_factor), int(self.h * self.scale_factor)))
            gt = cv2.resize(gt, (int(self.w * self.scale_factor), int(self.h * self.scale_factor)))
            img1 = cv2.resize(img1, (int(self.w * self.scale_factor), int(self.h * self.scale_factor)))
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
        # BGR -> RGB
        img0 = img0[:, :, ::-1]
        img1 = img1[:, :, ::-1]
        gt = gt[:, :, ::-1]
                
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, self.train_crop, self.train_crop)
            # temporal order reversal
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            # vertical flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            # horizontal flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            # 90 degree rotation
            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1) / 255.
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1) / 255.
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1) / 255.
        return [img0, gt, img1]
