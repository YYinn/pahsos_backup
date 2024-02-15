import os
from torch.functional import F
import torch.nn as nn
import torch
from torch import autograd
from torch.nn import *
from torchsummary import summary

class MSC_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MSC_block, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        
        self.batc1 = nn.BatchNorm3d(out_ch*2)
        self.act1 = nn.ReLU()
        self.conv3 = nn.Conv3d(out_ch * 2, out_ch, 3, padding=1)
        self.conv4 = nn.Conv3d(out_ch * 2, out_ch, 5, padding=2)

    def forward(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(input)
        cat1 = torch.cat([c1 , c2], dim=1)
        c3 = self.conv3(cat1)
        c4 = self.conv4(cat1)
        cat2 = torch.cat([c3, c4], dim=1)
        bat1 = self.batc1(cat2)
        re1 = self.act1(bat1)
        return re1

class Single16(nn.Module): #5 layers CNN branch
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super(Single16, self).__init__()
        # print('in_ch', in_ch, out_ch)
        self.cba1 = MSC_block(in_ch, out_ch)
        self.pool1 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba2 = MSC_block(out_ch*2, out_ch)
        self.pool2 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba3 = MSC_block(out_ch*2, out_ch)
        self.pool3 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba4 = MSC_block(out_ch*2, out_ch)
        self.pool4 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba5 = MSC_block(out_ch*2, out_ch)
        self.pool5 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.lin1 = nn.Linear(out_ch * 16, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(128, 1)
        self.re1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, input):
        cb1 = self.cba1(input)
        p1 = self.pool1(cb1)
        cb2 = self.cba2(p1)
        p2 = self.pool2(cb2)
        cb3 = self.cba3(p2)
        p3 = self.pool3(cb3)
        cb4 = self.cba4(p3)
        p4 = self.pool4(cb4)
        cb5 = self.cba5(p4)
        p5 = self.pool5(cb5)

        l = p5.view(p5.size(0), -1)
        l = self.dropout(l)
        l1 = self.lin1(l)
        r1 = self.re1(l1)

        r1 = self.dropout(r1)
        l2 = self.lin2(r1)
        r2 = self.re1(l2)

        r2 = self.dropout(r2)
        l3 = self.lin3(r1)
        s1 = self.sigmoid(l3)
        return s1



class mymodel(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(mymodel,self).__init__()

        self.single16 = Single16(1,out_ch=32)
        

    def forward(self, input):
       
        adc_output = self.single16(input)

        return adc_output


if __name__ == "__main__":
    model = mymodel()
    # set_trace()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,':',param.size())
    print(model)
    summary(model,(1 ,64, 64, 64))

