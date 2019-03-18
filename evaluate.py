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
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/separableConv') # Explicitly say which experiment you are performing

parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--cuda_avail', type=str, default='True')
parser.add_argument('--experimentNo', type=int, default= 0) # For different tuning Set this to the next consecutive number that have not previously setted
args = parser.parse_args()
Z_dim = 128
fixed_z = Variable(torch.randn(50000, Z_dim)).cuda()

def load_pretrained(model_type, cuda_avail):
    # This function loads the previously trained generator and discriminator for a given experimentNo
    print("Loading pretrained models for the experimentNo %s"%args.experimentNo)
    if args.model == "resnet":
        if args.cuda_avail == "True":
            discriminator = model_resnet.Discriminator().cuda()
            generator = model_resnet.Generator(Z_dim).cuda()
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo)))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo)))
        else:
            discriminator = model_resnet.Discriminator()
            generator = model_resnet.Generator(Z_dim)
            discriminator.load_state_dict(torch.load('checkpoints/disc_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load('checkpoints/gen_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))

    else:
        if args.cuda_avail == "True":
            discriminator = model.Discriminator().cuda()
            generator = model.SeparableGenerator2(Z_dim).cuda()
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo)))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo)))
        else:
            discriminator = model.Discriminator()
            generator = model.SeparableGenerator2(Z_dim)
            discriminator.load_state_dict(torch.load(args.checkpoint_dir + '/disc_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))
            generator.load_state_dict(torch.load(args.checkpoint_dir + '/gen_best' + str(args.experimentNo), map_location=lambda storage, loc: storage))

    print("The generator that has been loaded is %s"%generator)
    print("The discriminator that has been loaded is %s"%discriminator)
    return generator, discriminator

def inceptionScore(generator, inceptionModel):
    # This function assumed to be run during training to check the improvement in terms of InceptionScore
    # Therefore it directly assumes cuda is available during training
    batchSize = 100
    totalTrainingSamples = 50000 # By default Cifar-10
    samples = np.zeros((totalTrainingSamples,3,32,32), dtype = np.float32)
    for i in range(totalTrainingSamples // batchSize): #Get the predictions batch by batch
        samples[i*batchSize:(i+1)*batchSize] = generator(fixed_z[i*batchSize:(i+1)*batchSize]).cpu().detach().numpy()
    samples2 = np.array(((samples + 1) * 255/2.0).astype("uint8"), dtype=np.float32) # Conversion is important Scale between 0 and 255 generator last layer tanh()
    inceptionModel.to_gpu()
    print ("Calculating Inception Score...")
    mean, std = inception_score(inceptionModel,samples2,batch_size = 200 )
    print("inception score mean %s, std %s"%(mean, std))
    return mean, samples

######################################################################################################
#********************* Load the pretrained models ***************************************************#
#********************* Calculate the inception scores ***********************************************#
#********************* Show generated images ********************************************************#
######################################################################################################
def evaluate():
    generator, discriminator = load_pretrained(args.model, args.cuda_avail)
    inceptionModel = Inception()
    serializers.load_hdf5("model/inception_score.model",inceptionModel)
    inceptionModel.to_gpu()
    generator.eval()
    incepScore, samples = inceptionScore(generator, inceptionModel) # Calculate the inception score for each 10 epochs
 
    np.save("generated.npy", samples) # Save first 100 images

evaluate()
