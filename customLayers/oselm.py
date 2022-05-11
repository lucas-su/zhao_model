import torch.nn as nn
import torch.nn.functional as F
import torch


class OSELM(nn.Module):
    def __init__(self):
        super().__init__()
        # elm
        self.elmPooling = nn.AvgPool2d(16) # is 56 in paper but changed to 15 to accomodate smaller resnet input shape
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.elm1 = nn.Linear(40,1000)
        self.relu = torch.nn.LeakyReLU()
        self.elm2 = nn.Linear(1000, 10, bias=False)

    def forward(self, x):
        x = self.elmPooling(x)
        x = self.flatten(x)
        x = self.elm1(x)
        x = self.relu(x)
        x = self.elm2(x)
        return x