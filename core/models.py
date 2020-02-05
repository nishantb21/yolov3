import torch
from core.common import Convolution, Flatten, Dense, residual_repeater

class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.s1_output = None
        self.s2_output = None
        self.s3_output = None

    def get_s1_output(self):
        return self.s1_output    
    
    def get_s2_output(self):
        return self.s2_output
    
    def get_s3_output(self):
        return self.s3_output

class DarkNet53(Backbone):
    def __init__(self, channels, classes):
        super(DarkNet53, self).__init__()
        self.s1 = torch.nn.Sequential(Convolution(channels, 32, 3, 1), Convolution(32, 64, 3, 2), residual_repeater(1, 64, 32, 64), Convolution(64, 128, 3, 2), residual_repeater(2, 128, 64, 128), Convolution(128, 256, 3, 2), residual_repeater(8, 256, 128, 256))
        self.s1_output = 256
        self.s2 = torch.nn.Sequential(Convolution(256, 512, 3, 2), residual_repeater(8, 512, 256, 512))
        self.s2_output = 512
        self.s3 = torch.nn.Sequential(Convolution(512, 1024, 3, 2), residual_repeater(4, 1024, 512, 1024))
        self.s3_output = 1024
        self.classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), Flatten(), torch.nn.Linear(1024, classes))

    def forward(self, inp):
        x = self.s1(inp)
        x = self.s2(x)
        x = self.s3(x)
        x = self.classifier(x)
        return x

class yolov3(torch.nn.Module):
    def __init__(self, backbone, classes, scales):
        super(yolov3, self).__init__()
        output_shape = scales * (4 + 1 + classes)
        self.backbone = backbone
        self.block_1 = torch.nn.Sequential(Convolution(self.backbone.get_s3_output(), 512, 1, 1), Convolution(512, 1024, 3, 1), Convolution(1024, 512, 1, 1), Convolution(512, 1024, 3, 1), Convolution(1024, 512, 1, 1))
        self.large_boxes = torch.nn.Sequential(Convolution(512, 1024, 3, 1), Convolution(1024, output_shape, 1, 1, activation="linear"))
        self.block_2 = torch.nn.Sequential(Convolution(512, 256, 1, 1), torch.nn.Upsample(scale_factor=2))
        self.block_3 = torch.nn.Sequential(Convolution(self.backbone.get_s2_output() + 256, 256, 1, 1), Convolution(256, 512, 3, 1), Convolution(512, 256, 1, 1), Convolution(256, 512, 3, 1, 1), Convolution(512, 256, 1, 1))
        self.medium_boxes = torch.nn.Sequential(Convolution(256, 512, 3, 1), Convolution(512, output_shape, 1, 1, activation="linear"))
        self.block_4 = torch.nn.Sequential(Convolution(256, 128, 1, 1), torch.nn.Upsample(scale_factor=2))
        self.block_5 = torch.nn.Sequential(Convolution(self.backbone.get_s1_output() + 128, 128, 1, 1), Convolution(128, 256, 3, 1), Convolution(256, 128, 1, 1), Convolution(128, 256, 3, 1), Convolution(256, 128, 1, 1))
        self.small_boxes = torch.nn.Sequential(Convolution(128, 256, 3, 1), Convolution(256, output_shape, 1, 1, activation="linear"))

    def forward(self, inp):
        s1_op = self.backbone.s1(inp)
        s2_op = self.backbone.s2(s1_op)
        s3_op = self.backbone.s3(s2_op)

        block_1_output = self.block_1(s3_op)
        l_b = self.large_boxes(block_1_output)

        block_2_output = self.block_2(block_1_output)
        block_2_output = torch.cat([block_2_output, s2_op], axis=1)
        block_3_output = self.block_3(block_2_output)
        m_b = self.medium_boxes(block_3_output)

        block_4_output = self.block_4(block_3_output)
        block_4_output = torch.cat([block_4_output, s1_op], axis=1)
        block_5_output = self.block_5(block_4_output)
        s_b = self.small_boxes(block_5_output)

        return s_b, m_b, l_b

backbones = {}
backbones["DarkNet53"] = DarkNet53