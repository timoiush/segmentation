import torch
import torch.nn as nn
from blocks import *


class UNet0(nn.Module):
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
    
    
# U-Net with 2D/3D versions
class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,
            blocks=4,
            channels=32,
            activation='leaky',
            normalization='instance',
            dim=2,
            use_attention=False
            ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = blocks
        self.channels = channels
        self.activation = activation
        self.normalization = normalization
        self.dim = dim
        self.use_attention = use_attention
        self.pooling = pooling_layer(pooling='max', dim=self.dim)
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.attention_gates = []  #AttentionBlock(256, 128, 128)

        # Contracting path: encoder
        for i in range(self.blocks):
            in_ch = self.in_channels if i == 0 else out_ch
            out_ch = self.channels * (2 ** i)
            bottleneck = False if i < self.blocks - 1 else True
            encoder_block = EncoderBlock(
                in_ch,
                out_ch,
                activation=self.activation,
                normalization=self.normalization,
                dim=self.dim
            )
            self.encoder_blocks.append(encoder_block)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        # Expanding path: attention gates and decoder
        for i in range(self.blocks - 1):
            in_ch = out_ch
            out_ch = in_ch // 2
            if self.dim == 2:
                att_gate = AttentionBlock(in_ch, out_ch, out_ch)
            elif self.dim == 3:
                att_gate = AttentionBlock_3D(in_ch, out_ch, out_ch)
            self.attention_gates.append(att_gate)
            
            decoder_block = DecoderBlock(
                in_ch,
                out_ch,
                activation=self.activation,
                normalization=self.normalization,
                dim=self.dim)
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.conv = conv_layer(
            out_ch,
            self.out_channels,
            kernel_size=3,
            padding=1,
            dim=self.dim)

    def forward(self, x):
        encoder_outputs = []

        for i, block in enumerate(self.encoder_blocks):
            xs = block(x)
            if i < self.blocks -1:
                x = self.pooling(xs)
            else:
                x = xs
            encoder_outputs.append(xs)

        for i, block in enumerate(self.decoder_blocks):
            gate = self.attention_gates[i]
            skip_input = encoder_outputs[-(i + 2)]
            if self.use_attention:
                skip_input = gate(x, skip_input)
            x = block(x, skip_input)        
        return self.conv(x)
