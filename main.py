import os

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from resnet import *

learning_rate = 0.1
epoch_size = 200
batch_size = 128

loss_list = []
train_acc_list = []
test_acc_list = []
start_epoch = 0

# prepare dataset
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# build model
net = ResNet18()
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# load from checkpoint
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

if os.path.isfile('./checkpoint/ResNet.pth'):
    print("loading model...")
    checkpoint = torch.load('./checkpoint/ResNet.pth')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch'] + 1
    loss_list = checkpoint['loss_list']
    train_acc_list = checkpoint['train_acc_list']
    test_acc_list = checkpoint['test_acc_list']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# train and test
print("start training from epoch %d ..." % start_epoch)
for epoch in range(start_epoch):
    print("epoch: %d\ttrain loss: %.5f\t\ttest accuracy: %.2f%%" % (epoch, loss_list[epoch], test_acc_list[epoch] * 100))

for epoch in range(start_epoch, epoch_size):
    # train
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = correct/total
    train_loss = train_loss/(batch + 1)
    loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    # test
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct/total
    test_acc_list.append(test_acc)

    scheduler.step()

    print("epoch: %d\ttrain loss: %.5f\t\ttest accuracy: %.2f%%" % (epoch, train_loss, test_acc*100))

    # save training results
    # print("saving model...")
    state = {
        'net': net.state_dict(),
        'acc': test_acc_list[-1],
        'epoch': epoch,
        'loss_list': loss_list,
        'test_acc_list': test_acc_list,
        'train_acc_list': train_acc_list
    }
    torch.save(state, './checkpoint/ResNet18.pth')

# show training data
epoch = range(0, epoch_size)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(epoch, loss_list, '.-', color='blue', label='train loss')
ax1.set_ylim(-0.2, 2.4)
ax1.legend(loc=2)
ax2.plot(epoch, test_acc_list, '.-', color='red', label='test accuracy')
ax2.set_ylim(0, 1.2)
ax2.legend(loc=1)
plt.title("ResNet training result")

ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.show()