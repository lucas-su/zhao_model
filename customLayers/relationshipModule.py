import torch.nn as nn
import torch

class RelationshipAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        # relation
        self.fc1 = nn.Linear(2048,128)
        self.conv3 = nn.Conv2d(10,1,3, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout()

        self.att_w_c = torch.nn.Parameter(torch.zeros(1,1,32))
        self.att_b_c = torch.nn.Parameter(torch.ones(20,32,32))
        self.att_w_i = torch.nn.Parameter(torch.zeros(20,32,32))
        self.att_b_i = torch.nn.Parameter(torch.ones(20,32,32))

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.conv4 = nn.Conv2d(10,3,3,padding="same")
        self.conv5 = nn.Conv2d(10, 3, 3, padding="same")
        self.pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(10,10) # check

    def attention(self,x,y):

        ###################
        ### FC1 and FC2 in paper are attention line 1 and 2 (activations match)
        ### omega c output should be Bs x 10, and represents object labels
        ###################


        cat_in = torch.cat((x.unsqueeze(dim=4),y.unsqueeze(dim=4)), dim=4)
        cat_flat = self.flatten(cat_in)
        x = self.fc1(cat_flat)
        x = torch.kron(self.att_w_c, x) # output size should be BSx128x10
        x = torch.add(x, self.att_b_c) # output size should be BSx10
        x = torch.tanh(x)
        w = torch.softmax(self.att_w_i*x+self.att_b_i,dim=0)
        return w

    def forward(self, x):
        y = self.conv3(x)
        y = self.batchnorm1(y)
        y =  self.dropout(y)
        tiled = torch.tile(y, [10, 1 ,1])
        Omega_c = self.attention(x, tiled)

        ###
        # concatenation in attention leads to shape (batch_size, 20, height,width) from 2x (batch_size, 10, height,width)
        # after some operation, this is then multiplied with x with shape (batch_size, 10, height,width)
        ###

        y = torch.multiply(Omega_c, x)
        y = self.pool(y)
        y = self.fc(y)

        return y