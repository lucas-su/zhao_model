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

        self.att_w_c = torch.nn.Parameter(torch.ones(32,2))
        self.att_b_c = torch.nn.Parameter(torch.zeros(32,2))
        self.att_w_i = torch.nn.Parameter(torch.ones(32,1))
        self.att_b_i = torch.nn.Parameter(torch.zeros(10))

        self.flatten = torch.nn.Flatten(start_dim=2)
        self.conv4 = nn.Conv2d(10,3,3,padding="same")
        self.conv5 = nn.Conv2d(10, 3, 3, padding="same")
        # self.pool = nn.AvgPool2d(3)
        # self.fc = nn.Linear(10,10)
        self.upsample = torch.nn.Upsample(scale_factor=8) # 32*8=256

    def attention(self,x,y):

        ###################
        ### FC1 and FC2 in paper are attention line 1 and 2 (activations match)
        ### omega c output should be Bs x 10, and represents object labels
        ###################


        cat_in = torch.cat((x.unsqueeze(dim=4),y.unsqueeze(dim=4)), dim=4)
        # cat_flat = self.flatten(cat_in)

        x = torch.tensordot(cat_in, self.att_w_c) # ends up close to correct except it's BS, 10, 32 instead of BS, 10, 128

        # x = self.fc1(cat_flat)
        # x = torch.multiply(x, self.att_w_c) # output size should be BSx128x10, multiply is not correct here probably
        # x = torch.add(x, self.att_b_c)
        x = torch.tanh(x)
        # w = torch.matmul(x, torch.ones(10, 32).to("cuda").T)
        w = torch.matmul(x, self.att_w_i).squeeze() # take out singleton dimension
        # w = torch.multiply(self.att_w_i, x)
        w = torch.add(w, self.att_b_i)
        w = torch.softmax(w,dim=0) # output size should be BSx10
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


        return Omega_c