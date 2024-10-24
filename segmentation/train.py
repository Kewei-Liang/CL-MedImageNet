
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
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from monai.transforms import (  
    CastToTyped,
    AsDiscrete,
    ClassesToIndicesd,
    Compose,
    CropForegroundd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from monai.apps.auto3dseg.transforms import EnsureSameShaped

import logging
import sys
from monai.losses import DeepSupervisionLoss

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
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]
    max_epochs = 200
    val_interval = 1

    roi=[288, 288, 48]

    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, dtype=None, allow_missing_keys=True, image_only=True),
        EnsureTyped(keys=["image", "label"], data_type="tensor", dtype=torch.float, allow_missing_keys=True),
        EnsureSameShaped(keys="label", source_key="image", allow_missing_keys=True, warn=False),
        
        CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True, margin=10, allow_smaller=True),
        ScaleIntensityRanged(keys="image", a_min=-80, a_max=250, b_min=-1, b_max=1, clip=False),
        Lambdad(keys="image", func=lambda x: torch.sigmoid(x)),
        CastToTyped(keys="label", dtype=torch.uint8, allow_missing_keys=True),
        SpatialPadd(keys=["image", "label"], spatial_size=roi),
        ClassesToIndicesd(
                        keys="label",
                        num_classes=3,
                        indices_postfix="_cls_indices",
                        max_samples_per_class=2000,
                    ),
        RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="label",
                    num_classes=2,
                    spatial_size=roi,
                    num_samples=3,
                    ratios=None,
                    indices_key='label_cls_indices',
                    warn=False,
                ),
        RandAffined(keys=["image", "label"],
                        prob=0.2,
                        rotate_range=[0.26, 0.26, 0.26],
                        scale_range=[0.2, 0.2, 0.2],
                        mode=["bilinear", "nearest"],
                        spatial_size=roi,
                        cache_grid=True,
                        padding_mode="border",),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            
        RandGaussianSmoothd(
                keys="image", prob=0.2, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]),
        RandScaleIntensityd(keys="image", prob=0.5, factors=0.3),
        RandShiftIntensityd(keys="image", prob=0.5, offsets=0.1),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
            ]
    )

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

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1, num_workers=4)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    print("train len:{}".format(len(train_loader)))

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    print("val len:{}".format(len(val_loader)))
    device = torch.device("cuda:0")
    model=SegResNetDS(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=2,
        norm= "INSTANCE",
        blocks_down=[1, 2, 2, 4, 4],
        dsdepth=4,
        ).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_function = DeepSupervisionLoss(loss_function)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = WarmupCosineSchedule(
                optimizer=optimizer, warmup_steps=3, warmup_multiplier=0.1, t_total=max_epochs
            )
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    from tqdm import tqdm
    for epoch in range(max_epochs):
       
        logging.info('lr = {}'.format(lr_scheduler.get_lr()))
        logging.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
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

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join("./", "/best_seg_metric_model.pth"))
                    print("saved new best metric model")
               

                logging.info(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}")

        logging.info(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")


if __name__ == '__main__':
    main()


