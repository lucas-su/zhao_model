import torch.nn as nn
import torch.nn.functional as F
import torch


class OSELM(nn.Module):
    def __init__(self, dataset, norm, dropout, activation):
        super().__init__()
        if norm:
            self.use_norm = True
        else:
            self.use_norm = False
        if dropout:
            self.use_dropout = True
        else:
            self.use_dropout = False
        if activation:
            self.use_activation = True
        else:
            self.use_activation = False
        # elm
        self.flatten = torch.nn.Flatten(start_dim=1)
        # self.elm1 = nn.Linear(32, 1000)
        self.relu = torch.nn.LeakyReLU()
        if dataset == 'umd':
            self.elmPooling = nn.AvgPool2d(16) # is 56 in paper but changed to accomodate resnet input shape
            # self.elm1 = nn.Linear(272,1000)
            out_features = 17
            self.elm1 = nn.Linear(16*out_features, 1000)
        elif dataset == 'iitaff':
            self.elmPooling = nn.AvgPool2d(32)
            self.elm1 = nn.Linear(40, 1000)
            out_features = 10
        else:
            raise ValueError
        self.elm2 = nn.Linear(1000, out_features, bias=False)

    def forward(self, x):
        x = self.elmPooling(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.elm1(x)
        if self.use_activation:
            x = self.relu(x)
        x = self.elm2(x)

        return x