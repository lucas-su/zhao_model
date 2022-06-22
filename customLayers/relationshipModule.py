import torch.nn as nn
import torch

class RelationshipAwareModule(nn.Module):
    def __init__(self, dataset, norm, dropout, activation):
        super().__init__()
        if dataset == 'umd':
            self.nchannels = 17
        elif dataset == 'iitaff':
            self.nchannels = 10
        else:
            raise ValueError
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


        # relation
        # self.fc1 = nn.Linear(2048,128)
        self.conv3 = nn.Conv2d(self.nchannels, 1, 3, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout()

        n_intermediate_features = 128 # is 128 in paper

        self.att_w_c = nn.Parameter(torch.ones(self.nchannels, 8192, self.nchannels, n_intermediate_features))
        self.att_b_c = nn.Parameter(torch.zeros(self.nchannels, n_intermediate_features))
        self.att_w_i = nn.Parameter(torch.ones(self.nchannels, n_intermediate_features, self.nchannels))
        self.att_b_i = nn.Parameter(torch.zeros(self.nchannels))

        self.flatten = torch.nn.Flatten(start_dim=2)
        # self.conv4 = nn.Conv2d(10,3,3,padding="same")
        # self.conv5 = nn.Conv2d(10, 3, 3, padding="same")
        # self.pool = nn.AvgPool2d(3)
        # self.fc = nn.Linear(10,10)
        self.upsample = torch.nn.Upsample(scale_factor=8) # 32*8=256

    def attention(self,x,y):
        x = torch.cat((x.unsqueeze(dim=4),y.unsqueeze(dim=4)), dim=4)

        x = self.flatten(x)
        x = torch.tensordot(x, self.att_w_c)
        x = torch.add(x, self.att_b_c)
        x = torch.tanh(x) # output size should be BSx128x10
        w = torch.tensordot(x, self.att_w_i)
        w = torch.add(w, self.att_b_i)
        w = torch.softmax(w,dim=0) # output size should be BSx10
        return w

    def forward(self, x):
        y = self.conv3(x)
        if self.use_norm:
            y = self.batchnorm1(y)
        if self.use_dropout:
            y =  self.dropout(y)
        if self.use_activation:
            y = torch.relu(y)
        tiled = torch.tile(y, [self.nchannels, 1 , 1])
        Omega_c = self.attention(x, tiled)
        return Omega_c