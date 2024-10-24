
from monai.utils import first, set_determinism
from segresnet_ds import SegResNetDS
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader,  decollate_batch
from monai.config import print_config
import torch
import os
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from monai.transforms import (  
    CastToTyped,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    AsDiscreted
)
from monai.apps.auto3dseg.transforms import EnsureSameShaped
import logging
import sys


print_config()
set_determinism(seed=0)
data_dic={"train": ["1.nii.gz", "2.nii.gz","3.nii.gz","4.nii.gz","5.nii.gz","6.nii.gz",], 
          "test": ["7.nii.gz"], 
          "val": ["8.nii.gz"]}

def main():
    logging.basicConfig(filename="./"+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    root_dir="./"
    print(root_dir)
    data_dir = os.path.join(root_dir, "Task1")
    train_images=[]
    train_labels =[]
    test_images=[]
    test_labels =[]
    val_images=[]
    val_labels =[]
    images=glob.glob(data_dir+"/imagesTr/**.gz")
    for path in images:
        label_path=path.replace("imagesTr","maskTr")
        name=path.split("/")[-1]
        if name in data_dic["train"]:
            train_images.append(path)
            train_labels.append(label_path)
        if name in data_dic["test"]:
            test_images.append(path)
            test_labels.append(label_path)
        if name in data_dic["val"]:
            val_images.append(path)
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]
    roi=[288, 288, 48]
    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=True),
        EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float, allow_missing_keys=True),
        EnsureSameShaped(keys="label", source_key="image", allow_missing_keys=True, warn=False),
        
        CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True, margin=10, allow_smaller=True),
        ScaleIntensityRanged(keys="image", a_min=-80, a_max=250, b_min=-1, b_max=1, clip=False),
        Lambdad(keys="image", func=lambda x: torch.sigmoid(x)),
        CastToTyped(keys="label", dtype=torch.uint8, allow_missing_keys=True),
            ]
    )


    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    print("val len:{}".format(len(val_loader)))
    device = torch.device("cuda:0")
    model=SegResNetDS(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=3,
        norm= "INSTANCE",
        blocks_down=[1, 2, 2, 4, 4],
        dsdepth=4,
        ).to(device)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
  
    post_pred = Compose([AsDiscreted(keys="pred",argmax=True, to_onehot=2),KeepLargestConnectedComponentd(keys="pred", num_components=1,is_onehot=True, connectivity=3)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    from tqdm import tqdm

    model.eval()
    model.load_state_dict(torch.load("./best_seg_metric_model.pth"))
    with torch.no_grad():
        for val_data in tqdm(val_loader):
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = roi
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate().item()
        print("Metric on original image spacing: ", metric)
        dice_metric.reset()


if __name__ == '__main__':
    main()


