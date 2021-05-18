import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# load dataset
training = datasets.FashionMNIST(
    root='data', train=True, transform=transforms.ToTensor(), download=True
)
test = datasets.FashionMNIST(
    root='data', train=False, transform=transforms.ToTensor(), download=True
)

# visualize data
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
indices = torch.randint(len(training), (10,))

fig, ax = plt.subplots(1, 10, figsize=(20, 3))
for i in range(10):
    img, label = training[indices[i].item()]
    ax[i].imshow(img.squeeze(), cmap="gray")
    ax[i].set_title(labels_map[label])
    ax[i].set_axis_off()
plt.show()

# making custom dataset
class Fmnist(Dataset):
    def __init__(self, data, transform=True):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if transform is needed, apply it here.
        img, label = self.data[idx]
        if self.transform is True:
            img = img.squeeze()
        return img, label

# check if it works as intended
data = Fmnist(training, transform=True)
data_iter = iter(data)
d, l = next(data_iter)
plt.imshow(d, cmap='gray')
plt.title(labels_map[l])
plt.show()

# building DataLoader -> generating mini batches!
train_loader = DataLoader(dataset=data, batch_size=16, shuffle=True, num_workers=0)
train_loader_iter = iter(train_loader)

for i, data in enumerate(train_loader):
    if i == 10:
        break
    x, y = data
    print(f'{i}th data -----')
    print(x.shape)
    print(y)
    print()