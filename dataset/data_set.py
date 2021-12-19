import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, images, image_dir, mask_dir, transform=None, train=True, df=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.df = df
        self.isTrain = train
        if df is not None:  # 若df不为空
            self.images = self.df["img"]  # 读取指定图片的id
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if self.df is None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.df is None:  # 若df不为空
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0

        if self.transform is not None:
            if self.df is None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations['image']
                mask = augmentations['mask']
            else:
                augmentations = self.transform(image=image)
                image = augmentations['image']
        if self.df is None:
            return {"image": image, "mask": mask}
        else:
            return {"image": image,"imgId": self.images[index]}