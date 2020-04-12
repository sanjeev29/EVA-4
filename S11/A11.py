import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.r1 = self.res_block(in_channels=128, out_channels=128)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.r2 = self.res_block(in_channels=512, out_channels=512)

        self.pool = nn.MaxPool2d(4, 4)

        self.fcc = nn.Linear(512, 10)
    
    def res_block(self, in_channels, out_channels):     
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )


    def forward(self, x):
        # Prep layer (Output: 32x32x64)
        x = self.prep(x)

        # Layer 1 (Output: 16x16x128)
        x = self.conv1(x)
        r1 = self.r1(x)
        x = x + r1
        x = x + F.relu(x)

        # Layer 2 (Output: 8x8x256)
        x = self.conv2(x)

        # Layer 3 (Output: 4x4x512)
        x = self.conv3(x)
        r2 = self.r2(x)
        x = x + r2
        x = x + F.relu(x)

        # Maxpool with kernel size as 4
        x = self.pool(x) # (Output: 1x1x512)
        
        x = x.view(-1, 512)
        
        # FCC
        x = self.fcc(x) # (Output: 1x10)

        # Softmax
        x = F.softmax(x, dim=-1)

        return x
        
