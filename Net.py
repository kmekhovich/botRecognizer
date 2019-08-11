import torch
import torch.nn as nn
import torchvision


class Net(nn.Module):

    def __init__(self, output_dims):
        super(Net, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.fc = nn.Sequential(
            nn.Linear(2048, output_dims)
            # nn.ReLU(),
            # nn.Linear(512, 120),
            # nn.Softmax(1)
        )

    def forward(self, x):
        return self.net(x)
