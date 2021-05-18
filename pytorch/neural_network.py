import torch
import numpy as np
import torchsummary as ts
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# define hyperparams
learning_rate = 3e-4
batch_size = 64


# data preparation
training = datasets.FashionMNIST(
    root='data', train=True, transform=transforms.ToTensor(), download=True
)
test = datasets.FashionMNIST(
    root='data', train=False, transform=transforms.ToTensor(), download=True
)

train_dataloader = DataLoader(dataset=training, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

# build simple dense model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

model = NeuralNet()
ts.summary(model, (1, 28, 28))

# define loss and optim
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# define training loop
def train_loop(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # forward prop
        pred = model(x)
        loss = loss_fn(pred, y)
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(model, dataloader, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# train!
epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(model, train_dataloader, loss_fn, optimizer)
    test_loop(model, test_dataloader, loss_fn)
print("Done!")