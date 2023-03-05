import seaborn as sns
import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader
from utils import split_images, visualize_dataset, traning_loop, SegmentationDataset
from model import UNET
import segmentation_models_pytorch as smp

sns.set_style("whitegrid")

device = "cuda" if torch.cuda.is_available() else "cpu"


class DLoader:
    """Stores all the DataLoader objects for convenient access"""

    def __init__(
        self,
        img_src="data/landcover-ai",
    ):

        self.img_src = img_src
        images_list = list(glob.glob(os.path.join(img_src, "images", "*.tif")))

    def splitImages(self, image_size=2048):
        split_images(TARGET_SIZE=image_size)

    def returnDataLoader():

        transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.HueSaturationValue(40, 40, 30, p=1),
                        A.RandomBrightnessContrast(
                            p=1, brightness_limit=0.2, contrast_limit=0.5
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomRotate90(p=1),
                        A.HorizontalFlip(p=1),
                        A.RandomSizedCrop(
                            min_max_height=(248, 512), height=512, width=512, p=1
                        ),
                    ],
                    p=0.5,
                ),
            ]
        )

        train_set = SegmentationDataset(mode="train")
        train_dloader = DataLoader(train_set, batch_size=8, num_workers=2)

        # Preparing datasets and DataLoaders
        train_set = SegmentationDataset(mode="train", transforms=transforms, ratio=0.6)
        test_set = SegmentationDataset(mode="test")
        val_set = SegmentationDataset(mode="val", ratio=0.7)

        train_dloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
        test_dloader = DataLoader(test_set, batch_size=8, num_workers=2)
        val_dloader = DataLoader(val_set, batch_size=8, num_workers=2)

        return train_dloader, test_dloader, val_dloader
