# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RandCNN(nn.Module):
    def __init__(self):
        super(RandCNN, self).__init__()
        
        input_channels = 1 
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1)
        self.flatten = Flatten()
        final_conv_size = 64 * 9 * 6
        num_features = 256
        self.lin_layer = nn.Linear(final_conv_size, num_features)

    def forward(self, x, intermediate_layer=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        if intermediate_layer:
            # returned output has dimensions: batch_size x channel x height x width
            return x
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.lin_layer(x)
        return x
