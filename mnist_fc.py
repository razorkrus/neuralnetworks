# A full connect neuron network training on mnist dataset
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
                nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(in_dim, n_hidden_1),
                nn.BatchNorm1d(n_hidden_1),
                nn.ReLU(True))
        self.layer2 = nn.Sequential(
                nn.Linear(n_hidden_1, n_hidden_2),
                nn.BatchNorm1d(n_hidden_2),
                nn.ReLU(True))
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# Hyper parameter settings
batch_size = 64
learning_rate = 1e-2
num_epochs = 10

data_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

train_dataset = datasets.MNIST(root='./../data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./../data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = simpleNet(28 * 28, 300, 100, 10)
# model = Activation_Net(28 * 28, 300, 100, 10)
# model = Batch_Net(28 * 28, 300, 100, 10)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ==================== Model Training ====================
for epoch in range(num_epochs):
    t_start = time.time()
    for data in train_loader:
        img, label = data
        img = img.view(img.size(0), -1)

        out = model(img)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_end = time.time()
    t_elapsed = t_end - t_start
    
    print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}, time consumed: {t_elapsed:.2f} seconds')


# ==================== Model Testing ====================
model.eval()
eval_loss = 0
eval_acc = 0

t_start = time.time()
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
t_end = time.time()
t_elapsed = t_end - t_start

model_name = type(model).__name__
test_loss = eval_loss / len(test_dataset)
test_acc = eval_acc / len(test_dataset)


# ==================== Result Output ====================
print('=' * 80)
print(f'Model name: {model_name}')
print(f'Test Loss: {test_loss:.6f}, Acc: {test_acc:.6f}, train time: {t_elapsed:.2f} seconds')
print('=' * 80)




