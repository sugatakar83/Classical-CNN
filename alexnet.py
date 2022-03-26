import os
import numpy as np
import pandas as pd
import torch
# The torchvision package consists of popular datasets, model architectures,
# and common image transformations for computer vision.
import torchvision
import torch.nn as nn  # basic building blocks for graphs
import torch.optim as optim  # implementing various optimization algorithms
import matplotlib.pyplot as plt
import torch.nn.functional as F  # containing Convolution functions
from PIL import Image  # Read Image using Pillow
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

# creating the training environment
epochs = 10
batch_size = 128
device = ("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# create train and test dataset
train_csv = pd.read_csv('fashion-mnist_train.csv')
test_csv = pd.read_csv('fashion-mnist_test.csv')

# creating labeled dataset
class FashionDataset(Dataset):
    # call __init__ when object is created from the class and to initialize the attributes
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        label, image = [], []
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # −1 to indicate the length on the current axis needs to be automatically deduced according
        # to the rule that the total elements of the tensor remain unchanged
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')

    #  to returns the number of samples in our dataset
    def __len__(self):
        return len(self.images)

    # The __getitem__ function loads and returns a sample from the dataset at the given index idx
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform is not None:
            # transform the numpy array to PIL image
            pil_image = Image.fromarray(np.uint8(image))
            image  = self.transform(pil_image)
        return image, label


# input size of AlexNet is  227∗227 , and the image size of Fashion-MNIST is  28∗28 ,
# so we need to resize the image in the transform function
AlexTransform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# implement data loader to iterate through the data, manage batches, transform the data etc.
train_loader = DataLoader(
    FashionDataset(train_csv, transform=AlexTransform), batch_size=batch_size, shuffle=True)

test_loader = DataLoader(FashionDataset(test_csv, transform=AlexTransform), batch_size=batch_size, shuffle=True)

# # helper function to show an image
# def imageshow(img):
#     img = img.mean(dim=0)
#     img = img/2+0.5
#     npimg = img.numpy()
#     plt.imshow(npimg, cmap="Greys")
#
# # get some random images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# img_grid = torchvision.utils.make_grid(images[0]) # returns a tensor, containing the grid of the image
# # imageshow(img_grid)
# # print(class_names[labels[0]])
# # plt.show()


# creating the Alexnet layers
class f_mnist_alexnet(nn.Module):
    # create the constructor of f_mnist_alexnet class
    def __init__(self):
        # create a super class for accessing inherited methods that have been overridden in f_mnist_alexnet class
        super(f_mnist_alexnet, self).__init__()
        # create the convolution layers
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   )
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))

        # create the fully connected layers
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    # create the feed forward method
    def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            # If the number of last batch samples in the dataset is less than the defined batch_batch size,
            # a mismatch problem will occur. -1 will ensure the actual value for this dimension will be inferred
            # so that the number of elements in the view matches the original number of elements set in batch
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.dropout(out, 0.5) # 50% chance that the output of a given neuron will be forced to 0
            out = F.relu(self.fc2(out))
            out = F.dropout(out, 0.5)
            out = self.fc3(out)
            # The softmax activation function will return the probability that a sample represents a given image
            out = F.log_softmax(out, dim=1)
            return out


# create the model
model = f_mnist_alexnet().to(device)
criterion = F.nll_loss  # considering negative log likelihood loss as the training set having imbalanced classes
optimizer = optim.Adam(model.parameters())  # optimizing the model by using Adam algorithm


# method to train the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate (train_loader):
        target = target.type(torch.LongTensor)  # create the multi-dimentional matrix
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # setting the gradient as 0 before starting back propagationn
        output = model(data)
        loss = criterion(output, target) # calculate the loss for every epoch
        optimizer.step() # update the parameters based on loss calculation
        if(batch_idx+1)%30==0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# method to test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # disabling gradient calculation for test data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # keeping the output tensors are of the same size as input
            correct += pred.eq(target.view_as(pred)).sum().item() # operate on variables,need to be on the CPU again
            test_loss /= len(test_loader.dataset)
            print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
            print('=' * 50)


# train the model and then test it
for epoch in range(1,epochs+1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)





