import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model
from inception_score import inception_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--pretrained', type=str, default="False")
parser.add_argument('--cuda_avail', type=str, default="True")
args = parser.parse_args()
Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()

def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
def load_pretrained(model_type, cuda_avail):
    if args.model == "resnet":
        if args.cuda_avail == "True":
            discriminator = model_resnet.Discriminator().cuda()
            generator = model_resnet.Generator(Z_dim).cuda()
            discriminator.load_state_dict(torch.load('checkpoints/disc_last', map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load('checkpoints/gen_last', map_location=lambda storage, loc: storage))
        else:
            discriminator = model_resnet.Discriminator()
            generator = model_resnet.Generator(Z_dim)
            discriminator.load_state_dict(torch.load('checkpoints/disc_last', map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load('checkpoints/gen_last', map_location=lambda storage, loc: storage))

    else:
        if args.cuda_avail == "True":
            discriminator = model.Discriminator().cuda()
            generator = model.Generator(Z_dim).cuda()
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_last'))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_last'))
        else:
            discriminator = model.Discriminator()
            generator = model.Generator(Z_dim)
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_last', map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_last', map_location=lambda storage, loc: storage))

    return generator, discriminator        

os.makedirs(args.checkpoint_dir, exist_ok=True)
fixed_z = Variable(torch.randn(50000, Z_dim))
if args.pretrained == "True":
    generator, _ = load_pretrained(args.model, args.cuda_avail)
    samples = generator(fixed_z).cpu().data.numpy()
    print("Calculating inception_score ... mean %s, deviation %s"%(inception_score(samples, cuda = False if args.cuda_avail == "False" else True, batch_size = args.batch_size // 4, resize = True, splits = 10)))                

else:   # Here we assume cuda is a must for training 
    loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/', train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    if args.model == 'resnet':
        discriminator = model_resnet.Discriminator().cuda()
        generator = model_resnet.Generator(Z_dim).cuda()
    else:
        discriminator = model.Discriminator().cuda()
        generator = model.Generator(Z_dim).cuda()

    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
    fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
    for epoch in range(2000):
        train(epoch)
        evaluate(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
