"""
Bayesian 3D Residual UNet with Attention Gates + MC Dropout
for Pancreas Segmentation on Abdominal CT (MSD Task 07)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class BayesianUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32, dropout_p=0.3):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock3D(in_ch, base_ch)
        self.enc2 = ResidualBlock3D(base_ch, base_ch*2)
        self.enc3 = ResidualBlock3D(base_ch*2, base_ch*4)
        self.enc4 = ResidualBlock3D(base_ch*4, base_ch*8)
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_ch*8, base_ch*16)
        self.drop_bottleneck = nn.Dropout3d(p=dropout_p)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, 2, stride=2)
        self.att4 = AttentionGate3D(F_g=base_ch*8, F_l=base_ch*8, F_int=base_ch*4)
        self.dec4 = ResidualBlock3D(base_ch*16, base_ch*8)
        self.drop4 = nn.Dropout3d(p=dropout_p)
        
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.att3 = AttentionGate3D(F_g=base_ch*4, F_l=base_ch*4, F_int=base_ch*2)
        self.dec3 = ResidualBlock3D(base_ch*8, base_ch*4)
        self.drop3 = nn.Dropout3d(p=dropout_p)
        
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.att2 = AttentionGate3D(F_g=base_ch*2, F_l=base_ch*2, F_int=base_ch)
        self.dec2 = ResidualBlock3D(base_ch*4, base_ch*2)
        self.drop2 = nn.Dropout3d(p=dropout_p)
        
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.att1 = AttentionGate3D(F_g=base_ch, F_l=base_ch, F_int=base_ch//2)
        self.dec1 = ResidualBlock3D(base_ch*2, base_ch)
        self.drop1 = nn.Dropout3d(p=dropout_p)
        
        self.out = nn.Conv3d(base_ch, out_ch, 1)
        
    def forward(self, x, enable_dropout=False):
        # Encoder
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        
        # Bottleneck
        bn = self.bottleneck(self.pool(c4))
        if enable_dropout or self.training:
            bn = self.drop_bottleneck(bn)
        
        # Decoder with attention
        u4 = self.up4(bn)
        a4 = self.att4(g=u4, x=c4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))
        if enable_dropout or self.training:
            d4 = self.drop4(d4)
        
        u3 = self.up3(d4)
        a3 = self.att3(g=u3, x=c3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))
        if enable_dropout or self.training:
            d3 = self.drop3(d3)
        
        u2 = self.up2(d3)
        a2 = self.att2(g=u2, x=c2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))
        if enable_dropout or self.training:
            d2 = self.drop2(d2)
        
        u1 = self.up1(d2)
        a1 = self.att1(g=u1, x=c1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))
        if enable_dropout or self.training:
            d1 = self.drop1(d1)
        
        return self.out(d1)


if __name__ == "__main__":
    x = torch.randn(1, 1, 128, 128, 128)
    model = BayesianUNet3D(in_ch=1, out_ch=1)
    out = model(x, enable_dropout=True)
    print("Output shape:", out.shape)  # [1, 1, 128, 128, 128]
