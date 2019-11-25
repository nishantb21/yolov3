import torch
from common import Convolution, Flatten, Dense, residual_repeater

class DarkNet53(torch.nn.Module):
    def __init__(self, height, width, channels, classes):
        super(DarkNet53, self).__init__()
        downsampled_height = height // 32
        downsampled_width = width // 32

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
        self.avg_pool = torch.nn.AvgPool2d((downsampled_height, downsampled_width))
        self.flatten = Flatten()
        self.final_cls = Dense(1024, classes)

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

if __name__ == "__main__":
    height = 512
    width = 512
    batch_size = 8
    channels = 3
    classes = 1000

    x = torch.zeros((batch_size, channels, height, width))
    net = DarkNet53(height, width, channels, classes)
    y = net(x)
    print(y.shape)
    input()