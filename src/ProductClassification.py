import time

import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from ProductClassificationNet import Net
from common import utils
import time as timer

import torch.cuda.nccl as nccl
import torch.cuda


def imshow(img):
  ''' function to show image '''
  img = img.cpu()
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

if __name__ == '__main__':
    # https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212
    data_dir = utils.get_data_dir()
    # python image library of range [0, 1]
    # transform them to tensors of normalized range[-1, 1]

    transform = transforms.Compose(  # composing several transforms together
        [transforms.ToTensor(),  # to tensor object
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # mean = 0.5, std = 0.5

    # set batch_size
    batch_size = 4

    # set number of workers
    num_workers = 2

    # load train data
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    # load test data
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    # put 10 classes into a set
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get random training images with iter function
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # call function on our images
    imshow(torchvision.utils.make_grid(images))

    # print the class of the image
    print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))

    net = Net()
    print(net) # https://stats.stackexchange.com/questions/380996/convolutional-network-how-to-choose-output-channels-number-stride-and-padding/381032#381032

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    cuda_avail = torch.cuda.is_available()
    print(cuda_avail)
    if cuda_avail:
        dev = "cuda:0"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        dev = "cpu"
        start = timer.time()

    device = torch.device(dev)
    start.record()

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

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

    # save
    output_path = utils.get_output_dir()
    PATH = f'{output_path}/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    # reload
    net = Net()
    net.to(device)
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))