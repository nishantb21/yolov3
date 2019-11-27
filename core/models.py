import torch
from core.common import Convolution, Flatten, Dense, residual_repeater

class DarkNet53(torch.nn.Module):
    def __init__(self, channels, classes):
        super(DarkNet53, self).__init__()

        self.conv_1 = Convolution(channels, 32, 3, 1)
        self.conv_2 = Convolution(32, 64, 3, 2)
        self.repeater_1 = residual_repeater(1, 64, 32, 64)
        self.conv_3 = Convolution(64, 128, 3, 2)
        self.repeater_2 = residual_repeater(2, 128, 64, 128)
        self.conv_4 = Convolution(128, 256, 3, 2)
        self.repeater_3 = residual_repeater(8, 256, 128, 256)
        self.conv_5 = Convolution(256, 512, 3, 2)
        self.repeater_4 = residual_repeater(8, 512, 256, 512)
        self.conv_6 = Convolution(512, 1024, 3, 2)
        self.repeater_5 = residual_repeater(4, 1024, 512, 1024)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.final_cls = torch.nn.Linear(1024, classes)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.conv_2(x)
        x = self.repeater_1(x)
        x = self.conv_3(x)
        x = self.repeater_2(x)
        x = self.conv_4(x)
        x = self.repeater_3(x)
        x = self.conv_5(x)
        x = self.repeater_4(x)
        x = self.conv_6(x)
        x = self.repeater_5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.final_cls(x)

        return x

backbones = {}
backbones["DarkNet53"] = DarkNet53