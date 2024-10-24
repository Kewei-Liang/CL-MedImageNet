import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
import csv
import glob
import os
import numpy as np
import joblib
import random
from monai.utils import first, set_determinism
set_determinism(seed=1)
label_dice={}
all_dice={}
num=0

path=r'D:\labels_d.csv'
with open(path, mode='r', newline='', encoding='ISO-8859-1') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        if num>=1:
            name=row[0]
            label=int(row[1])-1
            label_dice[name]=label
            all_dice[name]=row
        num=num+1

train_labels=[]
train_images=[]
val_labels=[]
val_images=[]
test_labels=[]
test_images=[]

paths=glob.glob(r"D:\bianyuan_images\**.nii.gz")
images=[[] for _ in range(6)]
labels=[[] for _ in range(6)]
for path in paths:
    directory, name = os.path.split(path)
    name=name.replace(".nii.gz","")
    label=label_dice[name]
    images[label].append(name)

for i in range(6):
    n=len(images[i])
    random_numbers = random.sample(range(n), round(3*n/10))
    random_numbers2 = random.sample(random_numbers, round(len(random_numbers)/3))
    print("random_numbers",random_numbers)
    print("random_numbers2",random_numbers2)
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
print("counts_train",counts)
unique_elements, counts = np.unique(val_labels, return_counts=True)
print("counts_val",counts)
            

label_list=[]
sex_list=[]
age_list=[]

vol_list=[]
dx_list=[]
dy_list=[]
dz_list=[]

label_val_list=[]
sex_val_list=[]
age_val_list=[]

vol_val_list=[]
dx_val_list=[]
dy_val_list=[]
dz_val_list=[]

label_test_list=[]
sex_test_list=[]
age_test_list=[]

vol_test_list=[]
dx_test_list=[]
dy_test_list=[]
dz_test_list=[]
num=0
for row_path in train_images:
    row_name=row_path.replace(".nii.gz","").split("\\")[0]
    row=all_dice[row_name]
    
    label=int(row[1])-1
    sex=int(row[2])
    age=int(row[3])

    vol=float(row[4])
    dx=float(row[5])
    dy=float(row[6])
    dz=float(row[7])
    label_list.append(label)
    sex_list.append(sex)
    age_list.append(age)
    vol_list.append(vol)
    dx_list.append(dx)
    dy_list.append(dy)
    dz_list.append(dz)
for row_path in  val_images:
    row_name=row_path.replace(".nii.gz","").split("\\")[0]
    row=all_dice[row_name]
    
    label=int(row[1])-1

    sex=int(row[2])
    age=int(row[3])
   
    vol=float(row[4])
    dx=float(row[5])
    dy=float(row[6])
    dz=float(row[7])
    label_val_list.append(label)
    sex_val_list.append(sex)
    age_val_list.append(age)
    vol_val_list.append(vol)
    dx_val_list.append(dx)
    dy_val_list.append(dy)
    dz_val_list.append(dz)

for row_path in  test_images:
    row_name=row_path.replace(".nii.gz","").split("\\")[0]
    row=all_dice[row_name]
    
    label=int(row[1])-1
    sex=int(row[2])
    age=int(row[3])
    vol=float(row[4])
    dx=float(row[5])
    dy=float(row[6])
    dz=float(row[7])
    label_test_list.append(label)
    sex_test_list.append(sex)
    age_test_list.append(age)
    
    vol_test_list.append(vol)
    dx_test_list.append(dx)
    dy_test_list.append(dy)
    dz_test_list.append(dz)


# Sample data creation
train_data={}
train_data['label']=label_list
train_data['sex']=sex_list
train_data["age"]=age_list

train_data["volume"]=vol_list
train_data["dx"]=dx_list
train_data["dy"]=dy_list
train_data["dz"]=dz_list


val_data={}
val_data['label']=label_val_list
val_data['sex']=sex_val_list
val_data["age"]=age_val_list

val_data["volume"]=vol_val_list
val_data["dx"]=dx_val_list
val_data["dy"]=dy_val_list
val_data["dz"]=dz_val_list

test_data={}
test_data['label']=label_test_list
test_data['sex']=sex_test_list
test_data["age"]=age_test_list

test_data["volume"]=vol_test_list
test_data["dx"]=dx_test_list
test_data["dy"]=dy_test_list
test_data["dz"]=dz_test_list

df_train = pd.DataFrame(train_data)
df_val = pd.DataFrame(val_data)
df_test = pd.DataFrame(test_data)

X_train = df_train[['sex', 'age',"volume","dx","dy","dz"]]
y_train = df_train['label']

X_val = df_val[['sex', 'age',"volume","dx","dy","dz"]]
y_val = df_val['label']

X_test = df_test[['sex', 'age',"volume","dx","dy","dz"]]
y_val = df_val['label']
# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
data={}
for i in range(len(train_images)):
    data[train_images[i]]=str(X_train_scaled[i][0])+"_"+str(X_train_scaled[i][1])+"_"+str(X_train_scaled[i][2])+"_"+str(X_train_scaled[i][3])+"_"+str(X_train_scaled[i][4])+"_"+str(X_train_scaled[i][5])+"_"+str(X_train_scaled[i][6])

for i in range(len(val_images)):
    data[val_images[i]]=str(X_val_scaled[i][0])+"_"+str(X_val_scaled[i][1])+"_"+str(X_val_scaled[i][2])+"_"+str(X_val_scaled[i][3])+"_"+str(X_val_scaled[i][4])+"_"+str(X_val_scaled[i][5])+"_"+str(X_val_scaled[i][6])

for i in range(len(test_images)):
    data[test_images[i]]=str(X_test_scaled[i][0])+"_"+str(X_test_scaled[i][1])+"_"+str(X_test_scaled[i][2])+"_"+str(X_test_scaled[i][3])+"_"+str(X_test_scaled[i][4])+"_"+str(X_test_scaled[i][5])+"_"+str(X_test_scaled[i][6])

import json
filename="D:\linchuang.json"
with open(filename, 'w') as f:
    json.dump(data, f)  