import torch
from torch import nn
import torch.nn.functional as F

from spectral_normalization import SpectralNorm
from conditional_batch_norm import ConditionalBatchNorm2d
from self_attention import SelfAttentionPost, SelfAttention

channels = 3
leak = 0.1
w_g = 4
num_classes = 10


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = 'bilinear'

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class SeparableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(SeparableConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, (in_channels, 1), stride=1, padding=(1, 0))
        self.conv2 = nn.Conv2d(mid_channels, out_channels, (1, in_channels), stride=1, padding=(0, 1))

    def forward(self, x):
        u1 = torch.zeros((self.in_channels * 2, self.in_channels))
        u1[::2] = x
        u1 = u1.reshape(1, 1, u1.shape[0], u1.shape[1])
        res_conv1 = self.conv1(u1)

        u2 = torch.zeros((1, self.mid_channels, self.in_channels * 2, self.in_channels * 2))
        u2[:, :, ::2] = res_conv1.permute(0, 1, 3, 2)
        u2 = u2.permute(0, 1, 3, 2).reshape(1, self.mid_channels, u2.shape[2], u2.shape[3])
        res_conv2 = self.conv2(u2)

        return res_conv2


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1, 1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class Generator2(nn.Module):
    def __init__(self, z_dim):
        super(Generator2, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4, 4), mode='bilinear'),
            nn.Conv2d(z_dim, 512, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8, 8), mode='bilinear'),
            nn.Conv2d(512, 256, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16, 16), mode='bilinear'),
            nn.Conv2d(256, 128, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32, 32), mode=('bilinear')),
            nn.Conv2d(128, 64, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=1, padding=(1, 1)),  # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=None):
        super(SeparableConvBlock, self).__init__()
        groups = groups or in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=(1, 1),
                                   groups=groups)  # Each input channel is convolved Separately
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1,
                                   stride=1)  # Normal convolution with 1*1*in_channels kernels

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SeparableGenerator(nn.Module):
    def __init__(self, z_dim):
        super(SeparableGenerator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            Interpolate(size=(4, 4), mode='bilinear'),
            SeparableConvBlock(z_dim, 512),  # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8, 8), mode='bilinear'),
            SeparableConvBlock(512, 256),  # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16, 16), mode='bilinear'),
            SeparableConvBlock(256, 128),  # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32, 32), mode=('bilinear')),
            SeparableConvBlock(128, 64),  # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConvBlock(64, 3),  # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class SeparableGenerator2(nn.Module):
    def __init__(self, z_dim):
        super(SeparableGenerator2, self).__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(self.z_dim, self.z_dim * 4 * 4)  # Upsample first layer with fc layer

        self.model = nn.Sequential(
            SeparableConvBlock(z_dim, 512),  # Output is of size (batchNum,512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Interpolate(size=(8, 8), mode='bilinear'),
            SeparableConvBlock(512, 256),  # Output is of size (batchNum,256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Interpolate(size=(16, 16), mode='bilinear'),
            SeparableConvBlock(256, 128),  # Output is of size (batchNum,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Interpolate(size=(32, 32), mode=('bilinear')),
            SeparableConvBlock(128, 64),  # Output is of size (batchNum,64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConvBlock(64, 3),  # Output is of size (batchNum,3,32,32)
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.z_dim, 4, 4))


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


class SeparableSpectralNormalizedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, groups=None):
        super(SeparableSpectralNormalizedConvBlock, self).__init__()
        groups = groups or in_channels
        self.kernelSize = kernel
        self.stride = stride
        self.depthwise = SpectralNorm(
            nn.Conv2d(in_channels, in_channels, self.kernelSize, stride=self.stride, padding=(1, 1), groups=groups))  # Apply Spectral Norm and each input channel is convolved Separately
        self.pointwise = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, stride=1))  # Apply SpectralNorm and convolution with 1*1*in_channels kernels

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SeparableDiscriminator(nn.Module):
    def __init__(self):
        super(SeparableDiscriminator, self).__init__()

        self.conv1 = SeparableSpectralNormalizedConvBlock(channels, 64, 3, stride=1, groups=1)
        self.conv2 = SeparableSpectralNormalizedConvBlock(64, 64, 4, stride=2, groups=1)
        self.conv3 = SeparableSpectralNormalizedConvBlock(64, 128, 3, stride=1, groups=1)
        self.conv4 = SeparableSpectralNormalizedConvBlock(128, 128, 4, stride=2, groups=1)
        self.conv5 = SeparableSpectralNormalizedConvBlock(128, 256, 3, stride=1, groups=1)
        self.conv6 = SeparableSpectralNormalizedConvBlock(256, 256, 4, stride=2, groups=1)
        self.conv7 = SeparableSpectralNormalizedConvBlock(256, 512, 3, stride=1, groups=1)

        self.fc1 = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))

        return self.fc1(m.view(-1, w_g * w_g * 512))


class SAGenerator(nn.Module):
    def __init__(self, z_dim):
        super(SAGenerator, self).__init__()
        self.z_dim = z_dim

        self.conv1 = SpectralNorm(nn.ConvTranspose2d(z_dim, 512, 4, stride=1))
        self.bn1 = ConditionalBatchNorm2d(512, num_classes)
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)))
        self.bn2 = ConditionalBatchNorm2d(256, num_classes)
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)))
        self.bn3 = ConditionalBatchNorm2d(128, num_classes)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)))
        self.bn4 = ConditionalBatchNorm2d(64, num_classes)
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1, 1)))

    def forward(self, z, label):
        x = z.view(-1, self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x, label)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x, label)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x, label)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = self.bn4(x, label)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.Tanh()(x)

        return x


class SADiscriminator(nn.Module):
    def __init__(self):
        super(SADiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))

        self.attention_size = 16
        self.att = SelfAttention(128, self.attention_size)
        self.att_post = SelfAttentionPost(128, self.attention_size)

        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))

        self.embed = SpectralNorm(nn.Linear(num_classes, w_g * w_g * 512))

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x, c):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))

        self.attention_output = self.att(m)

        m = self.att_post(m, self.attention_output)

        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = m.view(-1, w_g * w_g * 512)

        return self.fc(m) + torch.bmm(m.view(-1, 1, w_g * w_g * 512), self.embed(c).view(-1, w_g * w_g * 512, 1))


'''
model = SeparableDiscriminator()
print("model that has been used is = ", model)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total params = ", pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable params = ", pytorch_total_params)
'''
