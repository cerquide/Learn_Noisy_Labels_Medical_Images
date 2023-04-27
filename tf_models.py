import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, img_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout=0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout=0.1)
        self.enc2 = conv_block(16, 32, dropout=0.1)
        self.enc3 = conv_block(32, 64, dropout=0.2)
        self.enc4 = conv_block(64, 128, dropout=0.2)

        self.middle = conv_block(128, 256, dropout=0.3)

        self.dec4 = conv_block(256, 128, dropout=0.2)
        self.dec3 = conv_block(128 , 64, dropout=0.2)
        self.dec2 = conv_block(64, 32, dropout=0.1)
        self.dec1 = conv_block(32, 16, dropout=0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        middle = self.middle(F.max_pool2d(enc4, 2))

        up4 = self.upconv4(middle)
        dec4 = self.dec4(torch.cat([up4, enc4], 1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], 1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], 1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], 1))

        return self.output(dec1)

def initialize_model(img_width, img_height, img_channels):
    model = UNet(img_channels)
    return model
