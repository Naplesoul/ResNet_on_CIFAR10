import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, 4 * out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * out_channels)

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(4 * out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet28(nn.Module):
    def __init__(self, label_num=10):
        super(ResNet28, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.new_basicblock_layer(64, 64, 1)
        self.layer2 = self.new_bottleneck_layer(64, 128, 2)
        self.layer3 = self.new_basicblock_layer(512, 256, 2)
        self.layer4 = self.new_bottleneck_layer(256, 512, 2)
        self.linear = nn.Linear(512 * 4, label_num)

    def new_bottleneck_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            Bottleneck(in_channels, out_channels, stride),
            Bottleneck(4 * out_channels, out_channels, 1),
            Bottleneck(4 * out_channels, out_channels, 1)
        )

    def new_basicblock_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, label_num=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.new_layer(64, 64, 1)
        self.layer2 = self.new_layer(64, 128, 2)
        self.layer3 = self.new_layer(128, 256, 2)
        self.layer4 = self.new_layer(256, 512, 2)
        self.linear = nn.Linear(512, label_num)

    def new_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet14(nn.Module):
    def __init__(self, label_num=10):
        super(ResNet14, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.new_layer(64, 64, 1)
        self.layer2 = self.new_layer(64, 128, 2)
        self.layer3 = self.new_layer(128, 256, 2)
        self.linear = nn.Linear(1024, label_num)

    def new_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 4)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
