# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:40:05 2023

@author: cakir
"""

import torch.nn as nn
import torch

class attention_block(nn.Module):
       def __init__(self, x_inchannel, gating_inchannel, inter_shape):
        super().__init__()
        self.phi_g = nn.Conv2d(gating_inchannel, inter_shape, kernel_size = 1, stride = 1, padding = 0)
        self.theta_x = nn.Conv2d(x_inchannel, inter_shape, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU()
        self.psi = nn.Conv2d(inter_shape, 1, kernel_size = 1, stride = 1, padding = 0)
        self.sig = nn.Sigmoid()

       def forward(self, x, gating):
        gating1 = self.phi_g(gating)
        x1 = self.theta_x(x)

        concat_xg = torch.add(gating1, x1)

        act_xg = self.relu(concat_xg)

        psi = self.psi(act_xg)

        sig_xg = self.sig(psi)

        return torch.mul(x, sig_xg)

def conv_block(ch_in,ch_out):
    conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    return conv

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.att5 = attention_block(512, 512, 256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.att4 = attention_block(256, 256, 128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.att3 = attention_block(128, 128, 64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.att2 = attention_block(64, 64, 32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x6 = self.att5(x4, d5)
        d5 = torch.cat((x6, d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x7 = self.att4(x3, d4)
        d4 = torch.cat((x7,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x8 = self.att3(x2, d3)
        d3 = torch.cat((x8,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x9 = self.att2(x1, d2)
        d2 = torch.cat((x9, d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1