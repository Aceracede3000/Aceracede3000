import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=10, layers=[2, 2, 2, 2], block=BasicBlock1D, dropout=True):
        super(ResNet1D, self).__init__()
        self.dropout = dropout
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(512, num_classes)
        self.out_activation = nn.Softmax()
        

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            if not self.dropout:
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout1d(p=0.5, inplace=False)
                )
                

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.dropout:
            x = nn.Dropout1d(p=0.5, inplace=False)(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out_activation(x)

        return x    
    
class AttResNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=10, layers=[2, 2, 2, 2], block=BasicBlock1D, dropout=True):
        super(AttResNet1D, self).__init__()
        self.dropout = dropout
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(512, num_classes)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.out_activation = nn.Softmax()
        

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            if not self.dropout:
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout1d(p=0.5, inplace=False)
                )
                

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, cond):
        attention_scores = self.attention(cond)
        attention_scores = attention_scores.unsqueeze(1)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        if self.dropout:
            x = nn.Dropout1d(p=0.5, inplace=False)(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out_activation(x)

        return x    
 
