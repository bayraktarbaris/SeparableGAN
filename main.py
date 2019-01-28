import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import model
import model_resnet
from incep_score_tf import inception_score, Inception
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from chainer import serializers
import chainer
import alternativeModels
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/Discriminator') # Explicitly say which experiment you are performing

parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--pretrained', type=str, default="False")
parser.add_argument('--cuda_avail', type=str, default="True")
parser.add_argument('--experimentNo', type=int, default= 0) # For different tuning Set this to the next consecutive number that have not previously setted
parser.add_argument('--disc_iters', type=int, default = 5) # This was the initial setup we trained this value should be cross validated
args = parser.parse_args()
Z_dim = 128
fixed_z = Variable(torch.randn(50000, Z_dim)).cuda()

def train(epoch):
    bestInceptionScore = 0
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(args.disc_iters):
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
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0]) # Show the loss for each 100 iteration
        if batch_idx % 1000 == 0:
            inceptionModel = Inception()
            serializers.load_hdf5("model/inception_score.model",inceptionModel)
            inceptionModel.to_gpu()
            generator.eval()
            incepScore = inceptionScore(generator, inceptionModel) # Calculate the inception score for each 10 epochs
            generator.train()
            if incepScore > bestInceptionScore: # Save the models as best generator and discriminator
                bestInceptionScore = incepScore
                torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format('best' + str(args.experimentNo)))) # Distinguish the experiments
                torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format('best' + str(args.experimentNo)))) # Distinguish the experiments
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
    # This function loads the previously trained generator and discriminator for a given experimentNo
    print("Loading pretrained models for the experimentNo %s"%args.experimentNo)
    if args.model == "resnet":
        if args.cuda_avail == "True":
            discriminator = model_resnet.Discriminator().cuda()
            generator = model_resnet.Generator(Z_dim).cuda()
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
        else:
            discriminator = model_resnet.Discriminator()
            generator = model_resnet.Generator(Z_dim)
            discriminator.load_state_dict(torch.load('checkpoints/disc_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load('checkpoints/gen_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))

    else:
        if args.cuda_avail == "True":
            discriminator = model.Discriminator().cuda()
            generator = model.Generator2(Z_dim).cuda()
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo)))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo)))
        else:
            discriminator = model.Discriminator()
            generator = model.Generator2(Z_dim)
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
    
    print("The generator that has been loaded is %s"%generator)
    return generator, discriminator        

def inceptionScore(generator, inceptionModel):
    # This function assumed to be run during training to check the improvement in terms of InceptionScore
    # Therefore it directly assumes cuda is available during training
    batchSize = 100
    totalTrainingSamples = 50000 # By default Cifar-10
    samples = np.zeros((totalTrainingSamples,3,32,32), dtype = np.float32)
    for i in range(totalTrainingSamples // batchSize): #Get the predictions batch by batch
        samples[i*batchSize:(i+1)*batchSize] = generator(fixed_z[i*batchSize:(i+1)*batchSize]).cpu().detach().numpy()
    samples = np.array(((samples + 1) * 255/2.0).astype("uint8"), dtype=np.float32) # Conversion is important Scale between 0 and 255 generator last layer tanh()
    inceptionModel.to_gpu()
    print ("Calculating Inception Score...")
    mean, std = inception_score(inceptionModel,samples,batch_size = 200 )
    print("inception score mean %s, std %s"%(mean, std))
    return mean
#os.makedirs(args.checkpoint_dir)# Do not forget ==> Python 2 has no argument exist_ok

#In order to just evaluate an experiment with given experimentNo set --pretrained=True
if args.pretrained == "True":
    samples = np.zeros((50000,3,32,32), dtype = np.float32)
    #generator, _ = load_pretrained(args.model, args.cuda_avail)
    #generator.eval()
    #for i in range(500):
        #samples[i*100:(i+1)*100]  = generator(fixed_z[i*100:(i+1)*100]).cpu().data.numpy()
    samples = np.load("generated.npy")
    samples = np.array(((samples + 1) * 255/2.0).astype("uint8"), dtype=np.float32) # Conversion is important Scale between 0 and 255
    model = Inception()
    serializers.load_hdf5("model/inception_score.model",model)
    model.to_gpu()
    print ("Calculating Inception Score...")
    #print("inception score mean %s, std %s"%(inception_score(model, np.array(cifar.train_data, dtype = np.float32).reshape((50000,3,32,32)))))
    print("inception score mean %s, std %s"%(inception_score(model,samples,batch_size = 200 )))

# pretrained=False we will train our networks
else:   # Here we assume cuda is a must for training 
    loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    if args.model == 'resnet':
        discriminator = model_resnet.Discriminator().cuda()
        generator = model_resnet.Generator(Z_dim).cuda()
    else:
        discriminator = model.SeperableDiscriminator().cuda()
        generator = model.Generator(Z_dim).cuda()

    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
    print("The generator currently trained is %s"%generator)
    print("The discriminator currently trained is %s"%discriminator)
    print("Current loss that is used is %s"%args.loss)
    print("Initial learning rate is %s"%args.lr)
    print("Number of times discriminator to be trained per generator training is %s"%args.disc_iters)
    print("All the models successfully loaded, Starting training... This will take a while")
    
    for epoch in range(2000):
        print("Training epoch %s"%epoch)
        train(epoch)
        
