# CL-MedImageNet

<p>
        <img width="850" src="https://github.com/Kewei-Liang/CL-MedImageNet/raw/main/fig1.png">
</p>
<br>

## <div align="center">Introduction</div>
<p>
  This is the python implementation of the paper "A Fully Automated Hybrid Approach for Predicting Sacral Tumor Types Using Deep Learning" and our paper will be published soon.
</p>

## <div align="center">Quick Start a Segmentation Example</div>
</details>
  
<details open>
<summary>Train with train.py</summary>

The segmentation folder stores training and evaluation codes for tumor segmentation.
Run segmentation/train.py for segmentation training.
```bash
python segmentation/train.py 
```
Run segmentation/val.py for segmentation evaluation.
```bash
python segmentation/val.py 
```
Hip segmentation code is similar to tumor segmentation code

## <div align="center">Quick Start a Classification Example</div>


“Normalize” file is used to normalize clinical information and location information
```bash
python Normalize.py 
```
Run train.py for classification training.
```bash
python train.py 
```
Run val.py for classification evaluation.
```bash
python val.py 
```
</details>
  
<details open>
</details>

