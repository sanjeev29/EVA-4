import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise_conv(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        dropout_rate = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 32x32x32 | Output: 32x32x32 | RF: 9x9

        self.pool1 = nn.MaxPool2d(2, 2) # Input: 32x32x32 | Output: 16x16x32 | RF: 10x10

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 16x16x32 | Output: 16x16x64 | RF: 18x18

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 16x16x64 | Output: 16x16x64 | RF: 26x26

        self.conv6 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels=64, out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 16x16x64 | Output: 16x16x64 | RF: 30x30

        self.pool2 = nn.MaxPool2d(2, 2) # Input: 16x16x64 | Output: 8x8x64 | RF: 32x32

        self.conv7 = nn.Sequential(
            DepthwiseSeparableConv2d(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 8x8x64 | Output: 8x8x128 | RF: 40x40

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ) # Input: 8x8x128 | Output: 8x8x128 | RF: 56x56

        self.pool3 = nn.MaxPool2d(2, 2) # Input: 8x8x128 | Output: 4x4x128 | RF: 60x60

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # Input: 4x4x128 | Output: 1x1x128 | RF: 60x60

        self.fc = nn.Linear(128, 10) # Input: 1x1x128 | Output: 1x1x10 | RF: 60x60



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.pool2(x)

        x = self.conv7(x)
        x = self.conv8(x)

        x = self.pool3(x)

        x = self.gap(x)
        x = x.view(-1, 128)

        x = self.fc(x)

        return x
