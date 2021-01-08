import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==================== Hyper Parameters ====================
batch_size = 16
learning_rate = 1e-2
num_epochs = 5

# ==================== Data Preparation ====================
DATA_DIR = './../data'
transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

train_dataset = datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True) 
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== Model Definition ====================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 10)
                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==================== Model Training ====================
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    t_start = time.time()
    for data in train_loader:
        img, label = data

        out = net(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item(): .6f}, time consumed: {t_elapsed:.2f} seconds')

# ==================== Model Testing ====================
net.eval()
running_loss = 0
running_acc = 0

t_start = time.time()
for data in test_loader:
    img, label = data

    out = net(img)
    loss = criterion(out, label)

    running_loss = loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    running_acc += num_correct.item()
t_end = time.time()
t_elapsed = t_end - t_start

test_loss = running_loss / len(test_dataset)
test_acc = running_acc / len(test_dataset)


# ==================== Print Result ====================
print('=' * 80)
print(f'Test Avg Loss: {test_loss:.6f}, Acc: {test_acc:.6f}, train time: {t_elapsed:.2f} seconds')
print('=' * 80)














