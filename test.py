import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def evaluate(epoch):
    samples = np.load("generated.npy")[:1000]
    # samples = ((samples + 1) * 255/2.0).astype("uint8")

    fig = plt.figure(figsize=(32, 32))
    gs = gridspec.GridSpec(32, 32)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1, 2, 0)), aspect='auto')

    if not os.path.exists('out/'):
        os.makedirs('out/')

    # plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    # plt.close(fig)
    plt.show()
    print(samples[0])


evaluate(10)
