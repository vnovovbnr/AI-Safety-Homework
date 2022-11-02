import torch
from torch import nn
from torch.functional import F


class Bottleneck(nn.Module):
    
    def __init__(self, inPlanes, outPlanes, stride = 1):
        super(Bottleneck,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inPlanes, outPlanes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outPlanes),
            nn.ReLU(),
            nn.Conv2d(outPlanes, outPlanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outPlanes)
        )
        self.shortCut = nn.Sequential()
        if inPlanes != outPlanes or stride > 1:
            self.shortCut = nn.Sequential(
                nn.Conv2d(inPlanes, outPlanes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outPlanes)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortCut(x)

        out = out1 + out2
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inPlanes = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = self._make_layer(Bottleneck, 64, 2, 2)
        self.layer2 = self._make_layer(Bottleneck, 128, 2, 2)
        self.layer3 = self._make_layer(Bottleneck, 256, 2, 2)
        self.layer4 = self._make_layer(Bottleneck, 512, 2, 2)

        self.fc = nn.Linear(512, 10)



    def _make_layer(self, block, planes, num, stride = 1):
        layers = []
        for i in range(num):
            if i ==0:
                in_stride = stride
            else:
                in_stride = 1
            layers.append(block(self.inPlanes, planes, in_stride))
            self.inPlanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x