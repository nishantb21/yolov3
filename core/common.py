import torch

class Dense(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation="linear"):
        super(Dense, self).__init__()
        self.dense = torch.nn.Linear(in_channels, out_channels)
        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "leaky":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "softmax":
            self.activation = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        x = self.dense(inp)
        x = self.activation(x)

        return x

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inp):
        return torch.flatten(inp, start_dim=1)

class Convolution(torch.nn.Module):
    ''' 
        Convolutional Module with Batchnormalzation and Activation
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding="same", activation="leaky"):
        super(Convolution, self).__init__()
        if padding == "same":
            padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "leaky":
            self.activation = torch.nn.LeakyReLU()

    def forward(self, inp):
        x = self.conv(inp)
        x = self.bn(x)
        x = self.activation(x)

        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, channels_1, channels_2):
        super(ResidualBlock, self).__init__()
        self.conv_1 = Convolution(in_channels, channels_1, 1, 1)
        self.conv_2 = Convolution(channels_1, channels_2, 3, 1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = self.conv_2(x)
        x += inp

        return x

def residual_repeater(number_repetitions, in_channels, channels_1, channels_2):
    layer = [ResidualBlock(in_channels, channels_1, channels_2) for i in range(number_repetitions)]
    return torch.nn.Sequential(*layer)