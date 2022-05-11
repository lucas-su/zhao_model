import torch.nn as nn
import torch

class RelationshipAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        # relation
        self.conv3 = nn.Conv2d(10,1,3, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout()

        self.att_w_c = torch.nn.Parameter(torch.zeros(20,32,32))
        self.att_b_c = torch.nn.Parameter(torch.ones(20,32,32))
        self.att_w_i = torch.nn.Parameter(torch.zeros(20,32,32))
        self.att_b_i = torch.nn.Parameter(torch.ones(20,32,32))

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc3 = nn.Linear(20480, 128)
        self.tanh = torch.nn.Tanh()
        self.fc4 = nn.Linear(128, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def attention(self,x,y):
        x = torch.multiply(self.att_w_c, torch.cat((x,y), dim=1))
        x = torch.add(x, self.att_b_c)
        x = torch.tanh(x)
        w = torch.softmax(self.att_w_i*x+self.att_b_i,dim=0)
        return w

    def forward(self, x):
        y = self.conv3(x)
        y = self.batchnorm1(y)
        y =  self.dropout(y)
        tiled = torch.tile(y ,[10 ,1 ,1])
        y = self.attention(x ,tiled)
        y = self.flatten(y)
        y = self.fc3(y)
        y = self.tanh(y)
        y = self.fc4(y)
        y = self.softmax(y)
        return y