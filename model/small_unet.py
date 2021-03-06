import torch
from torch import nn
import torch.nn.functional as F
# Define the network
# Then we define the network keeping in mind that down_1 output size should match with
# the scaled output of up_2, while down_2 size should match with the scaled output of the middle
# convolution of the u-net model.
class small_UNET_256(nn.Module):
    """
    Define UNET model that accepts a 256 input and mostly uses 3x3 kernels
    with stride and padding = 1. It reduces the size of the image to 8x8 pixels
    ** It might not work if the input 'x' is not a square.
    """

    def __init__(self):
        super(small_UNET_256, self).__init__()

        self.down_1 = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32, stride=2, padding=1))

        self.down_2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 128))

        self.middle = conv_block(128, 128, kernel_size=1, padding=0)

        self.up_2 = nn.Sequential(
            conv_block(256, 128),
            conv_block(128, 32))

        self.up_1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 32))

        self.output = nn.Sequential(
            conv_block(32, 16),
            conv_block(16, 1, kernel_size=1, padding=0))

    def forward(self, x):
        down1 = self.down_1(x)
        out = F.max_pool2d(down1, kernel_size=2, stride=2)

        down2 = self.down_2(out)
        out = F.max_pool2d(down2, kernel_size=2, stride=2)

        out = self.middle(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down2, out], 1)
        out = self.up_2(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down1, out], 1)
        out = self.up_1(out) + down1

        out = F.upsample(out, scale_factor=2)
        return self.output(out)


# Create the U-net model
# Let's create a smal u-net like model as a toy example.

# Define the convolutional block
# First we define the convolutional block so we do not need to re-write it every time
class conv_block(nn.Module):
    """
    Define the convolutional - batch norm - relu block to avoid re-writing it
    every time
    """

    def __init__(self, in_size, out_size, kernel_size=3, padding=1, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))