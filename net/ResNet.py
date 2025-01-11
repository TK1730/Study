import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity_x = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)

        out += identity_x
        return self.mish(out)

class ResidualLayer(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, block=BasicBlock):
        super(ResidualLayer, self).__init__()
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )

        self.first_block = block(in_channels, out_channels, downsample=downsample)
        self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))

    def forward(self, x):
        out = self.first_block(x)
        for block in self.blocks:
            out = block(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.mish = nn.Mish(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = ResidualLayer(2, in_channels=128, out_channels=128)
        self.layer2 = ResidualLayer(2, in_channels=128, out_channels=256)
        self.layer3 = ResidualLayer(2, in_channels=256, out_channels=512)
        self.layer4 = ResidualLayer(2, in_channels=512, out_channels=512)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 256, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.mish(out)
        out = self.fc2(out)

        return out
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        channel = channel // 4
        # 3x3 の畳み込み
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

        # skip connetion用のチャネル数調整
        self.shortcut = self._shortcut(channel, out_channels)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        shortcut = self.shortcut(x)
        y = self.relu(h + shortcut)
        return y

    def _shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return self._projection(in_channels, out_channels)
        else:
            return lambda x: x
        
    def _projection(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

if __name__ == '__main__':
    inputs = torch.zeros((16, 3, 227, 227))
    model = ResNet18(num_classes=10)
    outputs = model(inputs)
    print(outputs.size())

