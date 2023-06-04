#CIFAR 100 데이터 VGG모델

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.324, 0.292, 0.24), (0.784, 0.799, 0.838))
])

train_dataset = torchvision.datasets.CIFAR100(root='../../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR100(root='../../data/',
                                             train=False,
                                             transform=transforms.ToTensor())

#dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.maxpool = nn.MaxPool2d(2)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
    self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
    self.relu = nn.ReLU(inplace = True)
    self.fc1 = nn.Linear(8192, 2048)
    self.fc2 = nn.Linear(2048, 100)
  
  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.relu(self.conv5(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

model = VGG().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs):
  total_step = len(train_loader)
  curr_lr = learning_rate
  for i, (images, labels) in enumerate(train_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 ==0:
      print(num_epochs, i+1, total_step, loss.item())

def test():
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)   
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print("정확도", 100*correct/total)

for epoch in range(1, 1000):
  train(epoch)
  test()