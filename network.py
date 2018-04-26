import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Architecture of generator and discriminator networks are the same as Voice-GAN's
# https://arxiv.org/abs/1802.06840, accepted by IEEE ICASSP 2018
# https://github.com/Yolanda-Gao/VoiceGANmodel


class BottleneckEncoder(nn.Module):
    def __init__(self):
        super(BottleneckEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.norm4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.norm5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.norm6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, spec):
        relu1 = self.relu1(self.conv1(spec))
        relu2 = self.relu2(self.norm2(self.conv2(relu1)))
        relu3 = self.relu3(self.norm3(self.conv3(relu2)))
        relu4 = self.relu4(self.norm4(self.conv4(relu3)))
        relu5 = self.relu5(self.norm5(self.conv5(relu4)))
        relu6 = self.relu6(self.norm6(self.conv6(relu5)))

        return relu6, [relu1, relu2, relu3, relu4, relu5]


class BottleneckDecoder(nn.Module):
    def __init__(self):
        super(BottleneckDecoder, self).__init__()

        self.tconv6 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.tnorm6 = nn.BatchNorm2d(512)
        self.trelu6 = nn.ReLU(inplace=True)

        self.tconv5 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.tnorm5 = nn.BatchNorm2d(512)
        self.trelu5 = nn.ReLU(inplace=True)

        self.tconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.tnorm4 = nn.BatchNorm2d(256)
        self.trelu4 = nn.ReLU(inplace=True)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.tnorm3 = nn.BatchNorm2d(128)
        self.trelu3 = nn.ReLU(inplace=True)

        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.tnorm2 = nn.BatchNorm2d(64)
        self.trelu2 = nn.ReLU(inplace=True)

        self.tconv1 = nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False)

    def forward(self, feat):
        trelu6 = self.trelu6(self.tnorm6(self.tconv6(feat)))
        trelu5 = self.trelu5(self.tnorm5(self.tconv5(trelu6)))
        trelu4 = self.trelu4(self.tnorm4(self.tconv4(trelu5)))
        trelu3 = self.trelu3(self.tnorm3(self.tconv3(trelu4)))
        trelu2 = self.trelu2(self.tnorm2(self.tconv2(trelu3)))
        tconv1 = self.tconv1(trelu2)

        return torch.sigmoid(tconv1), [trelu2, trelu3, trelu4, trelu5, trelu6]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.network = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(32, 64, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(64 * 2),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(64 * 4),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(64 * 8),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.AdaptiveMaxPool2d(1),
                                     nn.Conv2d(512, 64, kernel_size=(1, 1)),
                                     nn.Conv2d(64, 8, kernel_size=(1, 1)),
                                     nn.Conv2d(8, 1, kernel_size=(1, 1)),
                                     nn.Sigmoid())

    def forward(self, spec):
        return self.network(spec)

# TODO: make branch for style discriminator test after debugging entire code
# TODO: implement style discriminator introduced in Voice-GAN, ICASSP 2018
# TODO: change training.py accordingly
# TODO: perform ablation test for style discriminator
