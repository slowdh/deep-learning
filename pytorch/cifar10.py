import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils import data
import torchvision.transforms as transforms
print(torch.__version__)


# transform image: normalize image to range of [-1, 1]
def normalize_img():
    transform = []
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform)

# create data loader
def data_loader(batch_size=64, is_train=True):
    train = torchvision.datasets.CIFAR10(root='./data', train=False, transform=normalize_img(), download=True)
    test = torchvision.datasets.CIFAR10(root='./data', train=False, transform=normalize_img(), download=True)
    dataset = train if is_train else test
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

# visualize imgs
visualize = False
if visualize:
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    sample_data = iter(data_loader(10, is_train=True))
    img, label = sample_data.next()
    img = (img / 2 + 0.5).numpy()

    fig, ax = plt.subplots(1, 10, figsize=(20, 3))
    for i in range(10):
        image = img[i]
        image = np.transpose(image, (1, 2, 0))
        ax[i].imshow(image)
        ax[i].set_axis_off()
        ax[i].set_title(classes[label[i].item()])
    plt.show()

# build model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # 32 * 16 * 16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 * 8 * 8
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 8 * 8, 120)
        self.linear2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def train(model, train_loader, loss_fn, optimizer, epoch, log_interval=100):
    for batch, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(data), len(train_loader.dataset),
                100. * batch / len(train_loader), loss.item()))

def test(model, test_loader, loss_fn):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# get dataloaders
train_loader = data_loader(64, is_train=True)
test_loader = data_loader(256, is_train=False)

# set optimizer and loss
model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# train!
epochs = 10
for epoch in range(1, epochs):
    train(model, train_loader, loss_fn, optimizer, epoch, log_interval=100)
    test(model, test_loader, loss_fn)
