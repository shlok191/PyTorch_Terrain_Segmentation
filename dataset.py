import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = {
    "train": "./data/landcover-ai/train.txt",
    "test": "./data/landcover-ai/test.txt",
    "validation": "./data/landcover-ai/val.txt",
}


class LandCoverDataset(Dataset):

    def __init__(self, split: str = "train"):
        """Initializes a DataLoader object with a particular data split

        Parameters
        ----------
        split : str, optional
           The split of the data to select, by default "train"
           Must be one of ["train", "test", "validation"]

        """

        super().__init__()

        # Check that the given data split is valid
        if split not in DATA_DIR.keys():

            raise ValueError(
                f"The given split value: {split} is not valid. It must be "
                + 'either "train", "test", or "validation"'
            )

        # Keeps track of the image iteration
        self.counter = 0
        self.img_name_file = DATA_DIR[split]

        # Read in all of the image names
        with open(self.img_name_file, "r") as file:
            file_names = file.read().splitlines()

        image_mask_pairs = []

        # Read in the images and their masks
        for name in file_names:

            image_path = f"./data/landcover-ai/output/{name}.jpg"
            mask_path = f"./data/landcover-ai/output/{name}_m.png"

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            image_mask_pairs.append((image, mask))

        self.image_mask_pairs = np.asarray(image_mask_pairs)

    def __len__(self) -> int:
        """Returns the number of image-mask pairs in the dataset split

        Returns
        -------
        int
            Image count for this split
        """
        return len(self.image_mask_pairs)

    def __getitem__(self, idx: int):
        """Returns an image-mask pair at the given index

        Parameters
        ----------
        idx : int
            The associated index
        """

        image, mask = self.image_mask_pairs[idx]

        # Convert to torch tensors and make the dimensions (C, H, W)!
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return (image, mask)
