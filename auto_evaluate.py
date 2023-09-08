#!/usr/bin/env python
# coding: utf-8

# In[59]:


import os
import re
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import torch
import albumentations as A
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


# In[20]:


IMAGE_PATH = '../segmentation/input/images/'
MASK_PATH = '../segmentation/input/masks/'
TARGET_SIZE = (800, 800)

def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df()
print('Total Images: ', len(df))


# In[46]:


def init_data(state):
    global test_set
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=state)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=state)
    
    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

    class DroneTestDataset(Dataset):
    
        def __init__(self, img_path, mask_path, X, transform=None):
            self.img_path = img_path
            self.mask_path = mask_path
            self.X = X
            self.transform = transform
    
        def __len__(self):
            return len(self.X)
    
        def __getitem__(self, idx):
            img = cv2.imread(self.img_path + self.X[idx] + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_path + self.X[idx] + '_label.png', cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('float64') / 255.0
    
            if self.transform is not None:
                aug = self.transform(image=img, mask=mask)
                img = Image.fromarray(aug['image'])
                mask = aug['mask']
    
            if self.transform is None:
                img = Image.fromarray(img)
    
            mask = torch.from_numpy(mask).long()
    
            return img, mask
    
    t_test = A.Resize(*TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    return DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)

def aIoU(pred_mask, mask, smooth=1e-10):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        true_class = (pred_mask == 1)
        true_label = (mask == 1)

        if true_label.long().sum().item() == 0: #no exist label in this loop
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()

            iou = (intersect + smooth) / (union +smooth)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)

def predict_image_mask_aiou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score = aIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score

def aiou_score(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_aiou(model, img, mask)
        score_iou.append(score)
    return score_iou


# In[66]:


for f in glob('*.pt'):
    if re.search('-t0\.\d\d\d', f):
        continue
    print(f)

    state = int(f.split('-')[2])
    if state == 521:
        state = 52024101
    test_set = init_data(state)
    
    MODEL_NAME = f.replace(".pt", "")
    model = torch.load(f)

    mob_aiou = aiou_score(model, test_set)
    mIoU = np.mean(mob_aiou)
    print('Test Set IoU: {:.3f}'.format(mIoU))

    os.rename(f, '.'.join(f.split('.')[:-1]) + f'-t{mIoU:.3f}.pt')


# In[63]:





# In[ ]:




