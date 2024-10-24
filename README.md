# CL-MedImageNet

<p>
    <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/Kewei-Liang/CL-MedImageNet/Fig1.tif"></a>
</p>
<br>

## <div align="center">Introduction</div>
<p>
  This is the python implementation of the paper "A Fully Automated Hybrid Approach for Predicting Sacral Tumor Types Using Deep Learning" and our paper will be published soon.
</p>

## <div align="center">Quick Start a Example</div>
</details>
  
<details open>
<summary>Train with train.py</summary>

`detect.py` runs train and saving results to `runs/train`.

```bash
python train.py  --data  data/JDC-MF-example.yaml
```

</details>

</details>
  
<details open>
<summary>Validation with val.py</summary>

`detect.py` runs validation and saving results to `runs/val`.

```bash
python val.py --data  data/JDC-MF-example.yaml
```

</details>

</details>
  
<details open>
<summary>Inference with detect.py</summary>

`detect.py` runs inference and saving results to `runs/detect`.

```bash
python detect.py 
```

</details>

