from torch import nn
import torch.nn.functional as F
import torch
from spectral_normalization import SpectralNorm

channels = 3
leak = 0.1
w_g = 4


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = 'bilinear'

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class ResidualBlock(nn.Module):
    # *******************  Residual Blocks with kernel size 7 doesnt seem to work somehow discriminator immediately converges and produces 0 loss *******************#
    def __init__(self, inputLayers):
        super(ResidualBlock, self).__init__()
        self.blockSizes = (inputLayers // 2, inputLayers // 8)
        self.block = nn.Conv2d
        self.inputLayers = inputLayers

    def forward(self, x):
        # first = self.block(self.inputLayers, self.blockSizes[0], 1, stride = 1).cuda()(x)
        second = self.block(self.inputLayers, self.blockSizes[0], 3, stride=1, padding=(1, 1)).cuda()(x)
        third = self.block(self.inputLayers, self.blockSizes[1], 5, stride=1, padding=(2, 2)).cuda()(x)
        x = torch.cat((second, third), 1)
        return self.block(self.blockSizes[0] + self.blockSizes[1], self.blockSizes[0] + self.blockSizes[1], 1,
                          stride=1).cuda()(x)


'''
class Generator3(nn.Module):
    def __init__(self, z_dim):
        super(Generator3, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4,4), mode='bilinear'),
            nn.Conv2d(z_dim,448,3,stride = 1, padding=(1,1)), # Output is of size (batchNum,448,4,4)
            nn.BatchNorm2d(448),
            nn.ReLU(),
            Interpolate(size=(8,8), mode='bilinear'),
            ResidualBlock(448,112), # Output is of size (batchNum,196,8,8)
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Interpolate(size=(16,16), mode='bilinear'),
            ResidualBlock(196,49), # Output is of size (batchNum,85,16,16)
            nn.BatchNorm2d(85),
            nn.ReLU(),
            Interpolate(size=(32,32), mode =('bilinear')),
            ResidualBlock(85,21), # Output is of size (batchNum,36,32,32)
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36,channels,3,stride=1, padding=(1,1)), # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Generator4(nn.Module):
    def __init__(self, z_dim):
        super(Generator4, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4,4), mode='bilinear'),
            nn.Conv2d(z_dim,512,3,stride = 1, padding=(1,1)), # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8,8), mode='bilinear'),
            ResidualBlock(512), # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16,16), mode='bilinear'),
            ResidualBlock(256), # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32,32), mode =('bilinear')),
            ResidualBlock(128), # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,channels,3,stride=1, padding=(1,1)), # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))
'''


class Generator5(nn.Module):
    def __init__(self, z_dim):
        super(Generator5, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4, 4), mode='bilinear'),
            nn.Conv2d(z_dim, 512, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8, 8), mode='bilinear'),
            ResidualBlock(512),  # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(320),
            nn.ReLU(),
            Interpolate(size=(16, 16), mode='bilinear'),
            nn.Conv2d(320, 200, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(200),
            nn.ReLU(),
            Interpolate(size=(32, 32), mode=('bilinear')),
            nn.Conv2d(200, 100, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 32, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, stride=1, padding=(1, 1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))

        return self.fc(m.view(-1, w_g * w_g * 512))
