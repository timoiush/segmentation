import torch
import torch.nn as nn
from models.blocks import *


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, channels=64):
        super().__init__()

        self.en1 = EncoderBlock(in_channels, channels)
        self.en2 = EncoderBlock(channels, channels*2)
        self.en3 = EncoderBlock(channels*2, channels*4)
        self.en4 = EncoderBlock(channels*4, channels*8)
        self.en5 = EncoderBlock(channels*8, channels*8)
        self.en6 = EncoderBlock(channels*8, channels*8)
        self.en7 = EncoderBlock(channels*8, channels*8)
        self.en8 = EncoderBlock(channels*8, channels*8, normalize=False)

        self.dec1 = DecoderBlock(channels*8, channels*8, dropout=0.5)
        self.dec2 = DecoderBlock(channels*16, channels*8, dropout=0.5)
        self.dec3 = DecoderBlock(channels*16, channels*8, dropout=0.5)
        self.dec4 = DecoderBlock(channels*16, channels*8)
        self.dec5 = DecoderBlock(channels*16, channels*4)
        self.dec6 = DecoderBlock(channels*8, channels*2)
        self.dec7 = DecoderBlock(channels*4, channels)
        self.last_layer = nn.Sequential(
            nn.ConvTranspose2d(channels*2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        e7 = self.en7(e6)
        e8 = self.en8(e7)
        d1 = self.dec1(e8, e7)
        d2 = self.dec2(d1, e6)
        d3 = self.dec3(d2, e5)
        d4 = self.dec4(d3, e4)
        d5 = self.dec5(d4, e3)
        d6 = self.dec6(d5, e2)
        d7 = self.dec7(d6, e1)

        return self.last_layer(d7)