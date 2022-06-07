import torch.nn as nn
import torch

class RelationshipAwareModule(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        if dataset == 'umd':
            self.out_features = 17
        elif dataset == 'iitaff':
            self.out_features = 10
        else:
            raise ValueError
        # relation
        # self.fc1 = nn.Linear(2048,128)
        self.conv3 = nn.Conv2d(self.out_features,1,3, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(1) # batch norm 1???
        self.dropout = nn.Dropout()

        self.att_w_c = nn.Parameter(torch.ones(self.out_features,30752,self.out_features,128)) # from 2048 changed after input resize
        self.att_b_c = nn.Parameter(torch.zeros(self.out_features,128))
        self.att_w_i = nn.Parameter(torch.ones(self.out_features, 128,self.out_features))
        self.att_b_i = nn.Parameter(torch.zeros(self.out_features))

        self.flatten = torch.nn.Flatten(start_dim=2)
        # self.conv4 = nn.Conv2d(10,3,3,padding="same")
        # self.conv5 = nn.Conv2d(10, 3, 3, padding="same")
        # self.pool = nn.AvgPool2d(3)
        # self.fc = nn.Linear(10,10)
        self.upsample = torch.nn.Upsample(scale_factor=8) # 32*8=256

    def attention(self,x,y):

        ###################
        ### FC1 and FC2 in paper are attention line 1 and 2 (activations match)
        ### omega c output should be Bs x 10, and represents object labels
        ###################

        cat_in = torch.cat((x.unsqueeze(dim=4),y.unsqueeze(dim=4)), dim=4)

        ###
        ## first flatten and then multiplying with weight with same shape works
        ## but is not consistent with method in paper
        cat_flat = self.flatten(cat_in)
        # x = self.fc1(cat_flat)
        # x = torch.multiply(x, self.att_w_c) # output size should be BSx128x10, multiply is not correct here probably
        ###


        ###
        # ends up close to correct except it's BS, 10, 32 instead of BS, 10, 128
        # image width and height is different than in paper (32 vs 112) because of dcnn and original image size as well
        # but this doesn't explain how they got to 128 from 112
        x = torch.tensordot(cat_flat, self.att_w_c)
        ###

        x = torch.add(x, self.att_b_c)

        x = torch.tanh(x) # output size should be BSx128x10
        # w = torch.matmul(x, torch.ones(10, 32).to("cuda").T)

        ###
        # unclear which multiply to use: matmul, mm, tensordot or (maybe) kron/outer?
        w = torch.tensordot(x, self.att_w_i)

        w = torch.add(w, self.att_b_i)
        w = torch.softmax(w,dim=0) # output size should be BSx10
        return w

    def forward(self, x):
        y = self.conv3(x)
        y = self.batchnorm1(y)
        y =  self.dropout(y)
        tiled = torch.tile(y, [self.out_features, 1 ,1])
        Omega_c = self.attention(x, tiled)

        return Omega_c