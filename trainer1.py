import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn.functional as F

# TODO 这个模型看起来还不错，可以改改看

# VAL_IMG_DIR =
# VAL_MASK_DIR =


import torch
import torch.nn as nn

# def crop_img(tensor, target_tensor):
#     target_size = target_tensor.size()[2]
#     tensor_size = tensor.size()[2]
#     delta = tensor_size-target_size
#     delta = delta//2
#     # all batch, all channels, heightModified,widthModified

#     return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


# if __name__ == "__main__":
#     image = torch.rand((3, 3, 572, 572))
#     model = UNet()
#     print(image.shape)
#     model(image)
import config
from config import TRAIN_IMG_DIR, TRAIN_MASK_DIR, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, SPLIT, BATCH_SIZE, LEARNING_RATE, \
    EPOCHS
from dataset.data_set import CarvanaDataset
from metrics import LogNLLLoss
from model.small_unet import small_UNET_256
from model.unet2 import UNet
from model.unet_cbam import UNet_cbam
from norm import dice_coef, iou_score
from rle2mask import files_mask2rle_single_from_np

images = os.listdir(TRAIN_IMG_DIR)
masks = os.listdir(TRAIN_MASK_DIR)

img = np.array(Image.open(TRAIN_IMG_DIR + "/" + images[0]).convert("RGB"))
plt.imshow(img, cmap="gray")
print(img.shape)

msk = np.array(Image.open(TRAIN_MASK_DIR + "/" + images[0].replace(".jpg", "_mask.gif")).convert("L"))
plt.imshow(msk, cmap="gray")
print(msk.shape)


def fit(model, dataloader, data, optimizer, criterion, learnRate=None):
    model.train()
    train_running_loss = 0.0
    counter = 0
    # num of batches
    num_batches = int(len(data) / dataloader.batch_size)
    # set a progress bar
    pbar = tqdm(enumerate(dataloader), total=num_batches)
    for i, data in pbar:
        counter += 1
        image, mask = data["image"].to(DEVICE), data["mask"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(image)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, mask)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            '[Training] - EPOCH %d / %d |learning Rate %.8f| avg LOSS: %.8f'
            % (epoch + 1, config.EPOCHS, learnRate, train_running_loss / counter))
    train_loss = train_running_loss / counter
    return train_loss


def validate(model, dataloader, data, criterion):
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    dice = 0
    iou = 0
    # number of batches
    num_batches = int(len(data) / dataloader.batch_size)
    pbar = tqdm(enumerate(dataloader), total=num_batches)
    with torch.no_grad():
        for i, data in pbar:
            counter += 1
            image, mask = data["image"].to(DEVICE), data["mask"].to(DEVICE)
            outputs = model(image)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, mask)
            dice += dice_coef(outputs, mask)
            iou += iou_score(outputs, mask)
            valid_running_loss += loss.item()
            pbar.set_description(
                '[Validating] - EPOCH %d / %d | BATCH LOSS: %.8f | current Dice : %.8f | current MIOU: %.8f'
                % (epoch + 1, config.EPOCHS, valid_running_loss / counter, dice / counter, iou / counter))
    valid_loss = valid_running_loss / counter
    return valid_loss


import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Rotate(limit=35, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0
    ),
    ToTensorV2()
])

validation_transform = A.Compose([
    A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])

normal_transform = A.Compose([
    A.Resize(1280, 1918),
])


def train_test_split(images, splitSize):
    imageLen = len(images)
    val_len = int(splitSize * imageLen)
    train_len = imageLen - val_len
    train_images, val_images = images[:train_len], images[train_len:]
    return train_images, val_images


train_images_path, val_images_path = train_test_split(images, SPLIT)
df = pd.read_csv("sample_submission.csv")
train_data = CarvanaDataset(train_images_path, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform, True)
valid_data = CarvanaDataset(val_images_path, TRAIN_IMG_DIR, TRAIN_MASK_DIR, validation_transform, True)
test_data = CarvanaDataset(val_images_path, config.TRAIN_TEST_DIR, TRAIN_MASK_DIR, validation_transform, True, df=df)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class DiceLoss(nn.Module):
    """损失函数"""

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


if __name__ == "__main__":
    name = "cbam_down1"
    # MODE = "train"
    MODE = "predict"
    # MODE = "test"
    model = UNet_cbam().to(DEVICE)
    learnRate = LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    criterion = nn.BCEWithLogitsLoss()
    model_dir = "exp/" + name + "_model.pth"
    if MODE == 'predict':
        b = os.path.exists(model_dir)
        if b:
            try:
                checkpoint = torch.load(model_dir)
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch'] + 1
            except:
                epoch = 0

            data = train_data.__getitem__(50)
            plt.imshow(data["image"].cpu().detach().numpy().transpose((1, 2, 0)))
            plt.show()
            plt.imshow(data['mask'], cmap="gray")
            plt.show()
                # print(train_data.__getitem__(0)['mask'].shape)

                # for Testing on Single datapoint after training
                # plt.imshow(np.transpose(np.array(data['image']),(1,2,0)),cmap="gray")
                # print(data['image'].shape)
            img = data['image'].unsqueeze(0).to(device="cuda")
                # model = UNet()
            output = model(img)
            output = torch.squeeze(output)
            output[output > 0.0] = 1.0
            output[output <= 0.0] = 0
                # print(torch.max(output))
                # print(output.shape)
            disp = output.detach().cpu()
            plt.imshow(disp, cmap="gray")
            plt.show()
    elif MODE == "test":
        b = os.path.exists(model_dir)
        if b:
            try:
                print("初始化模型")
                checkpoint = torch.load(model_dir)
                model.load_state_dict(checkpoint['model_state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch'] + 1
            except:
                epoch = 0
            num_batches = int(len(test_data) / test_dataloader.batch_size)
            # set a progress bar
            pbar = tqdm(enumerate(test_dataloader), total=num_batches)
            model.eval()
            valid_running_loss = 0.0
            counter = 0
            dice = 0
            iou = 0
            csv = open('submission.csv', 'w')
            # number of batches
            with torch.no_grad():
                for i, data in pbar:
                    counter += 1
                    image = data["image"].to(DEVICE)
                    # print(data["imgId"])
                    outputs = model(image)
                    # outputs = outputs.squeeze(1)
                    output = torch.squeeze(outputs)
                    output[output > 0.0] = 1.0
                    output[output <= 0.0] = 0
                    for j in range(BATCH_SIZE):
                        b = outputs[j].cpu().detach().numpy()
                        a = b.transpose((1, 2, 0))
                        c = normal_transform(image=a)['image']
                        # print(data["imgId"][j])
                        # files_mask2rle_single_from_np('submission.csv', data["imgId"][j], c)
                        b1 = data["image"][j].cpu().detach().numpy().transpose((1, 2, 0))
                        c1 = normal_transform(image=b1)['image']
                        plt.imshow(c1, cmap="gray")
                        plt.show()
                        plt.imshow(c, cmap="gray")
                        plt.show()

    else:
        train_loss = []
        val_loss = []
        best_loss = 0.0075
        for epoch in range(EPOCHS):
            if epoch == 0:
                b = os.path.exists(model_dir)
                if b:
                    try:
                        checkpoint = torch.load(model_dir)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        learnRate = checkpoint['learning_rate']
                        epoch = checkpoint['epoch'] + 1
                        # Todo 加入这个
                        best_loss = checkpoint['loss']
                        print(f"加载模型")
                    except:
                        print(f"加载模型失败，重置完成")
                        epoch = 0
            print(f"Epoch {epoch + 1} of {EPOCHS}")

            if epoch != 0 and ((epoch+1) % config.INTERVAL == 0):
                print('降低学习率')
                learnRate = learnRate * config.DECAY
                for params in optimizer.param_groups:
                    params['lr'] = learnRate
            train_epoch_loss = fit(model, train_dataloader, train_data, optimizer, criterion, learnRate=learnRate)
            val_epoch_loss = validate(model, valid_dataloader, valid_data, criterion)
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f'Val Loss: {val_epoch_loss:.4f}')
            if best_loss > val_epoch_loss:  # 模型好才加入
                print('更新最优网络')
                best_loss = val_epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': learnRate,
                    'loss': val_epoch_loss
                    # 'optimizer_state_dict': optimizer.state_dict(),
                }, model_dir)

        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color="orange", label='train loss')
        plt.plot(val_loss, color="red", label='validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        # plt.savefig(f"../input/loss.png")
        plt.show()

        print("\n---------DONE TRAINING----------\n")
