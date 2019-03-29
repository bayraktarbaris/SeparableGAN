import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    # Number of generated images
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("You have a CUDA device, set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = inception_model(x)
        return F.softmax(input=x, dim=1).data.cpu().numpy()

    # Get predictions, Inception network is trained on dataset with 1000 classes
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []
    scores = np.empty((splits), dtype=np.float32)
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        kl = part * (np.log(part) -
                     np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores[k] = np.exp(kl)

    return np.mean(scores), np.std(scores)


##################################################################################################################
#########################   This is used for testing inception score on original Cifar10 images     ##############
##################################################################################################################
if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __getNumberOfItems__(self, howMany):
            data = torch.zeros([howMany, 3, 32, 32], dtype=torch.float32)
            for i in range(howMany - 1):
                data[i] = self.__getitem__(i)
            return data

        def __len__(self):
            return len(self.orig)


    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='/slow_data/datasets/cifar-10-python', download=False,
                         transform=transforms.Compose([
                             transforms.Scale(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )

    IgnoreLabelDataset(cifar)

    print("Calculating Inception Score...")
    print("cifar.size = ", IgnoreLabelDataset(cifar).__getNumberOfItems__(500).size())
print(inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=80, resize=True, splits=10))
