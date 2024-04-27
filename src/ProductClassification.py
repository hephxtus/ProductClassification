import os
import time

import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for transformation
import pandas as pd
from skimage import io

import torch  # PyTorch package
import torchvision  # load datasets
import torchvision.transforms as transforms  # transform data
import torch.nn as nn  # basic building block for neural neteorks
import torch.nn.functional as F  # import convolution functions like Relu
import torch.optim as optim  # optimzer
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose

from ProductClassificationNet import Net, Net2
from common import utils
import time as timer

import torch.cuda.nccl as nccl
import torch.cuda
from torch.utils.data import Dataset

from DataSet import CustomDataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def imshow(img, title=None):
    ''' function to show image '''
    img = img.cpu()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def train(root_dir, PATH, cuda_avail=True, transform: v2.Compose = None ):
    # load train data

    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, padding=4),
        v2.RandomRotation(degrees=90),
        transform
    ])
    # train_transform = v2.Compose([
    #     v2.RandomResizedCrop(size=20, antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.RandomRotation(degrees=90),
    #     v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    #     transform,
    #
    #     # AddGaussianNoise(0,1),
    # ])
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True,
                                            download=True, transform=train_transform)
    print(trainset)
    # trainset = CustomDataset(root_dir=os.path.join(root_dir, 'train'), transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, drop_last=True)

    # put 10 classes into a set
    classes = trainset.classes

    print(classes)

    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # call function on our images
    imshow(torchvision.utils.make_grid(images), title=' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
    print(images.size())
    print(labels)
    # print the class of the image

    net = Net(num_classes=len(classes))
    print(
        net)  # https://stats.stackexchange.com/questions/380996/convolutional-network-how-to-choose-output-channels-number-stride-and-padding/381032#381032

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    if cuda_avail:
        dev = "cuda:0"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        net.cuda()
        criterion.cuda()
    else:
        dev = "cpu"
        start = timer.time()

    device = torch.device(dev)
    # net.to(device)
    if cuda_avail:
        start.record()

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            print(i, '/', len(trainloader))
            inputs, labels = data
            print(len(labels))
            print(labels)
            print(inputs.size())
            # zero the parameter gradients
            optimizer.zero_grad()

            if cuda_avail:
                inputs, labels = images.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            print(i, '/', len(trainloader))

    # whatever you are timing goes here
    print('Finished Training')
    if cuda_avail:
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(start.elapsed_time(end))  # milliseconds
    else:
        end = timer.time()
        print(end - start)

    torch.save(net.state_dict(), PATH)


def test(root_dir, PATH, cuda_avail=True, transform: v2.Compose=None):
    # load test data
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False,
                                           download=True, transform=transform)
    # testset = CustomDataset(root_dir=os.path.join(root_dir, 'test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    classes = testset.classes
    # reload

    # device = torch.device(dev)
    net = Net(num_classes=len(classes))
    # net.to(device)
    if cuda_avail:
        dev = "cuda:0"
        net.cuda()
    else:
        dev = "cpu"
    print(PATH)
    net.load_state_dict(torch.load(PATH))

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    if cuda_avail:
        images, labels = images.cuda(), labels.cuda()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    print(predicted)
    print('Predicted: ', ' '.join('%s' % classes[predicted[j]]
                                  for j in range(4)))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if cuda_avail:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d test images: %d %%' % (total,
            100 * correct / total))


if __name__ == '__main__':
    # https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
    data_dir = utils.get_data_dir()
    # python image library of range [0, 1]
    # transform them to tensors of normalized range[-1, 1]

    transform: v2.Compose = v2.Compose(  # composing several transforms together
        [
           v2.Resize((32, 32)),
           v2.ToTensor(),  # to tensor object
           v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])  # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    cuda_avail = torch.cuda.is_available()
    cuda_avail = False

    print(cuda_avail)

    root_dir = os.path.join(data_dir, 'cifar-10')

    # save
    output_path = utils.get_output_dir()
    PATH = f'{output_path}/product_class_net.pth'
    train(root_dir, PATH, cuda_avail, transform)
    test(root_dir, PATH, cuda_avail, transform)
