from torch import nn
import torch.nn.functional as F

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

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class Generator2(nn.Module):
    def __init__(self, z_dim):
        super(Generator2, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4,4), mode='bilinear'),
            nn.Conv2d(z_dim,512,3,stride=1,padding=(1,1)), # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8,8), mode='bilinear'),
            nn.Conv2d(512,256,3,stride=1,padding=(1,1)), # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16,16), mode='bilinear'),
            nn.Conv2d(256,128,3,stride=1,padding=(1,1)), # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32,32), mode =('bilinear')),
            nn.Conv2d(128,64,3,stride=1,padding=(1,1)), # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,channels,3,stride=1, padding=(1,1)), # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class SeperableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeperableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride = 1, padding=(1,1), groups = in_channels) # Each input channel is convolved seperately
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride = 1) # Normal convolution with 1*1*in_channels kernels
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SeperableGenerator(nn.Module):
    def __init__(self, z_dim):
        super(SeperableGenerator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4,4), mode='bilinear'),
            SeperableConvBlock(z_dim, 512), # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8,8), mode='bilinear'),
            SeperableConvBlock(512,256), # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16,16), mode='bilinear'),
            SeperableConvBlock(256,128), # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32,32), mode =('bilinear')),
            SeperableConvBlock(128,64), # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeperableConvBlock(64,3), # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class SeperableGenerator2(nn.Module):
    def __init__(self, z_dim):
        super(SeperableGenerator2, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, self.z_dim*4*4) # Upsample first layer with fc layer        

        self.model = nn.Sequential(
            SeperableConvBlock(z_dim, 512), # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8,8), mode='bilinear'),
            SeperableConvBlock(512,256), # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16,16), mode='bilinear'),
            SeperableConvBlock(256,128), # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32,32), mode =('bilinear')),
            SeperableConvBlock(128,64), # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeperableConvBlock(64,3), # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.z_dim, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


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

        return self.fc(m.view(-1,w_g * w_g * 512))

class SeperableSpectralNormalizedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(SeperableSpectralNormalizedConvBlock, self).__init__()
        self.kernelSize = kernel
        self.stride = stride
        self.depthwise = SpectralNorm(nn.Conv2d(in_channels, in_channels, self.kernelSize, stride = self.stride, padding=(1,1), groups = in_channels)) # Apply Spectral Norm and each input channel is convolved seperately
        self.pointwise = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, stride = 1)) # Apply SpectralNorm and convolution with 1*1*in_channels kernels
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SeperableDiscriminator(nn.Module):
    def __init__(self):
        super(SeperableDiscriminator, self).__init__()

        self.conv1 = SeperableSpectralNormalizedConvBlock(channels, 64, 3, stride = 1)

        self.conv2 = SeperableSpectralNormalizedConvBlock(64, 64, 4, stride=2)
        self.conv3 = SeperableSpectralNormalizedConvBlock(64, 128, 3, stride=1)
        self.conv4 = SeperableSpectralNormalizedConvBlock(128, 128, 4, stride=2)
        self.conv5 = SeperableSpectralNormalizedConvBlock(128, 256, 3, stride=1)
        self.conv6 = SeperableSpectralNormalizedConvBlock(256, 256, 4, stride=2)
        self.conv7 = SeperableSpectralNormalizedConvBlock(256, 512, 3, stride=1)


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

        return self.fc(m.view(-1,w_g * w_g * 512))

'''
model = SeperableGenerator2(128)
print("model that has been used is = ", model)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total params = ", pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable params = ", pytorch_total_params)
'''
