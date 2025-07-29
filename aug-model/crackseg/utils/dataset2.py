import os

import numpy as np
# from crackseg.utils.general import Augmentation
from crackseg.utils import augmentation
from PIL import Image
import cv2
from torch.utils import data
import torch
from torchvision.transforms import functional as F, ColorJitter, RandomErasing


class CrackAugment(data.Dataset):
    def __init__(
        self, root: str, suffix_images: str, suffix_masks: str, image_size: int = 240
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.path_images = os.path.join(root, suffix_images)
        self.path_masks = os.path.join(root, suffix_masks)
        self.filenames = list(set(os.listdir(self.path_images)) & set(os.listdir(self.path_masks)))
        self.color_transform = ColorJitter(contrast=(0.25, 1), saturation=(0, 2), hue=(-.5,0.5))
        self.occlude_transform = RandomErasing()
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        # image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        # mask_path = os.path.join(self.root, f"masks{os.sep}{filename + self.mask_suffix}.jpg")

        # image load
        image = np.asarray(Image.open(os.path.join(self.path_images, filename))) / 255.
        mask = cv2.imread(os.path.join(self.path_masks, filename), 0) / 255.

        # mask = np.asarray(Image.open(os.path.join(self.path_masks, filename)))
        # mask = (mask / 255.)

        # TODO: The mask must be binary. In `Road Crack` dataset the mask image has values between 0 and 255, however
        #  it was supposed to be 0 and 1. So mask image divided by 255 to make it between 0 and 1.
        # if (np.asarray(mask) > 1).any():
        #     mask = np.asarray(np.asarray(mask) / 255, dtype=np.byte)
        #     mask = Image.fromarray(mask)
        # assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        # resize
        # if self.transforms is not None:
        #     image, mask = self.transforms(image, mask)

        # image, mask = augmentation.random_crop_image(image, mask, (1200, 1200))
        image, mask = augmentation.weighted_crop(image, mask, (1000, 1000))
        image = augmentation.apply_gradient([image], grad_min=-0.6, grad_max=0.1)[0]

        image, mask  = augmentation.transform_perspective_image(image, mask)
        image, mask = augmentation.resize_image(image, mask, self.image_size, self.image_size)

        image = torch.as_tensor(image, dtype=torch.float).permute(2, 0, 1)
        mask = torch.as_tensor(mask, dtype=torch.int64)#.permute(2, 0, 1)
        image = self.color_transform.forward(image)

        i, j, h, w, v = self.occlude_transform.get_params(image, scale=(0, 0.33), ratio=(0.3, 3.3))
        image[:, i:i+h, j:j+w] = 0
        mask[i:i+h, j:j+w] = 0

        return image, mask
