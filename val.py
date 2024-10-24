from monai.utils import first, set_determinism
import logging
import os
import sys
import torch
import numpy as np
import monai
from monai.config import print_config
from monai.data import DataLoader
from dataset import ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    Resize,
    ScaleIntensity,
    Orientation,
    RandFlip,
    RandRotate,
    RandZoom
)
from sklearn import metrics
from tqdm import tqdm
from clmedimagenet import Clmedimagenet121
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

set_determinism(seed=1)

import csv
import random
label_dice={}
num=0
with open(r'D:\labels_d.csv', mode='r', newline='', encoding='ISO-8859-1') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if num>=1:
            name=row[0]
            label=int(row[1])-1
            label_dice[name]=label
        num=num+1

def main():
    logging.basicConfig(filename=r"D:\log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    print_config()
    root_dir=r"D:\bianyuan_images"
    print(root_dir)
    import csv
    import glob
    train_labels=[]
    train_images=[]
    test_labels=[]
    test_images=[]
    val_labels=[]
    val_images=[]

    paths=glob.glob(root_dir+"/**.nii.gz")
    images=[[],[],[],[],[],[]]
    for path in paths:
        directory, name = os.path.split(path)
        name=name.replace(".nii.gz","")
        label=label_dice[name]
        images[label].append(path)
    
    for i in range(6):
        n=len(images[i])
        random_numbers = random.sample(range(n), round(3*n/10))
        random_numbers2 = random.sample(random_numbers, round(len(random_numbers)/3))
        print(random_numbers)
        print(random_numbers2)
        label=i
        for j in range(n):
            if j  in random_numbers2:
                val_images.append(images[i][j])
                val_labels.append(label)
            elif j in random_numbers:
                test_images.append(images[i][j])
                test_labels.append(label)
            else:
                train_images.append(images[i][j])
                train_labels.append(label)
    unique_elements, counts = np.unique(train_labels, return_counts=True)
    print(counts)
    unique_elements, counts = np.unique(val_labels, return_counts=True)
    print(counts)
    unique_elements, counts = np.unique(test_labels, return_counts=True)
    print(counts)

    train_labels = torch.nn.functional.one_hot(torch.as_tensor(train_labels)).float()
    val_labels = torch.nn.functional.one_hot(torch.as_tensor(val_labels)).float()

    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),Orientation(axcodes="LPS"), Resize((256,256, 64)),RandRotate(range_x=np.pi / 12, range_y=np.pi / 12,range_z=np.pi / 12,prob=0.5, keep_size=True),RandFlip(spatial_axis=[0,1,2], prob=0.5),RandZoom(min_zoom=0.9, max_zoom=1, prob=0.5),])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(),Orientation(axcodes="LPS"), Resize((256, 256, 64))])

    train_ds = ImageDataset(image_files=train_images, labels=train_labels, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0, pin_memory=pin_memory)
    print("train:",len(train_loader))

    val_ds = ImageDataset(image_files=val_images, labels=val_labels, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=5, num_workers=0, pin_memory=pin_memory)
    print("test:",len(val_loader))
    model = Clmedimagenet121(spatial_dims=3, in_channels=1, out_channels=6).to(device)
    metric_values = []
    pred_list=[]
    pred_pr_list=[]
    label_list=[]

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score

    model.eval()
    model.load_state_dict(torch.load(r"D:\best_metric_model_classification3d_array.pth"))

    num_correct = 0.0
    metric_count = 0
    
    for val_data in tqdm(val_loader):
        val_images, val_labels,linchuang = val_data[0].to(device), val_data[1].to(device),val_data[2].to(device)
        with torch.no_grad():
            val_outputs = model(val_images,linchuang )
            pred_pr=np.array(F.softmax(val_outputs, dim=1).cpu())
            pred=np.array(val_outputs.argmax(dim=1).cpu())
            label= np.array(val_labels.argmax(dim=1).cpu())
            for j in range(len(pred)):
                pred_list.append(pred[j])
            for j in range(len(pred_pr)):
                pred_pr_list.append(pred_pr[j])
                label_list.append(label[j])
            value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
            metric_count += len(value)
            num_correct += value.sum().item()
    
    auc = metrics.roc_auc_score(label_list, pred_pr_list,multi_class='ovr')
    metric=auc
    metric_values.append(metric)

    micro-f1 = f1_score(label_list, pred_list,average='micro')
    macro-f1 = f1_score(label_list, pred_list,average='macro')

    C2 = confusion_matrix(label_list,pred_list,labels=[0,1,2,3,4,5])
           
           

if __name__ == '__main__':
    main()

