import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import torch
import os
import torchvision
import requests

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QuantAlexNet(AlexNet):
  def __init__(self):
    super(QuantAlexNet, self).__init__()
    self.quant = QuantStub()
    self.dequant = DeQuantStub()
  def forward(self,x):
    x=self.quant(x)
    x=super(QuantAlexNet,self).forward(x)
    x=self.dequant(x)
    return x

def QAlexnet(destination_path="/content/data/"):
    url="https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    r = requests.get(url)
    model_file=destination_path+'alexnet-owt-4df8aa71.pth'

    with open(model_file, 'wb') as f:
        f.write(r.content)

    model = QuantAlexNet()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    
    return model.to('cpu')
