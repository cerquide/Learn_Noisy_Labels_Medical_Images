import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, img_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout = 0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout = 0.1)
        self.enc2 = conv_block(16, 32, dropout = 0.1)
        self.enc3 = conv_block(32, 64, dropout = 0.2)
        self.enc4 = conv_block(64, 128, dropout = 0.2)

        self.middle = conv_block(128, 256, dropout = 0.3)

        self.dec4 = conv_block(256, 128, dropout = 0.2)
        self.dec3 = conv_block(128 , 64, dropout = 0.2)
        self.dec2 = conv_block(64, 32, dropout = 0.1)
        self.dec1 = conv_block(32, 16, dropout = 0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.output = nn.Conv2d(16, 1, kernel_size = 1)

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

### Global CM model ###

class gcm_layers(nn.Module):

    def __init__(self, input_width, input_height):
        super(gcm_layers, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.global_weights = nn.Parameter(torch.eye(2))

    def forward(self, x):

        all_weights = self.global_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        all_weights = all_weights.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, self.input_width, self.input_height)
        
        y = nn.functional.softmax(all_weights, dim = 1)

        return y
    
class GCM_UNet(nn.Module):
    def __init__(self, img_width, img_height, img_channels):
        super(GCM_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout = 0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout = 0.1)
        self.enc2 = conv_block(16, 32, dropout = 0.1)
        self.enc3 = conv_block(32, 64, dropout = 0.2)
        self.enc4 = conv_block(64, 128, dropout = 0.2)

        self.middle = conv_block(128, 256, dropout = 0.3)

        self.dec4 = conv_block(256, 128, dropout = 0.2)
        self.dec3 = conv_block(128 , 64, dropout = 0.2)
        self.dec2 = conv_block(64, 32, dropout = 0.1)
        self.dec1 = conv_block(32, 16, dropout = 0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.cms_output = nn.ModuleList()
        for i in range(3):
            self.cms_output.append(gcm_layers(img_width, img_height))

        self.output = nn.Conv2d(16, 1, kernel_size = 1, bias = True)

    def forward(self, x):

        self.output_cms = []

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

        for i in range(3):
            
            self.output_cms.append(self.cms_output[i](x))

        return self.output(dec1), self.output_cms

def initialize_model_GCM(img_width, img_height, img_channels):
    model = GCM_UNet(img_width, img_height, img_channels)
    return model



### local CM model ###

class lcm_layers(nn.Module):

    def __init__(self, in_channels):
        super(lcm_layers, self).__init__()

        def conv_block(in_channels, out_channels, dropout = 0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True)
            )
        
        self.class_no = 2

        self.conv_1 = conv_block(in_channels = in_channels, out_channels = in_channels)
        self.conv_2 = conv_block(in_channels = in_channels, out_channels = in_channels)
        self.conv_last = nn.Conv2d(in_channels, 4, kernel_size = 1, bias = True)
        # self.relu = nn.Softplus()
        self.relu = nn.functional.softmax(self.conv_last, dim = 1)

    def forward(self, x):

        y = self.relu(self.conv_last(self.conv_2(self.conv_1(x))))

        return y
    
class lCM_UNet(nn.Module):
    def __init__(self, img_channels):
        super(lCM_UNet, self).__init__()

        def conv_block(in_channels, out_channels, dropout = 0.0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace = True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, stride = 2)
            )

        self.enc1 = conv_block(img_channels, 16, dropout = 0.1)
        self.enc2 = conv_block(16, 32, dropout = 0.1)
        self.enc3 = conv_block(32, 64, dropout = 0.2)
        self.enc4 = conv_block(64, 128, dropout = 0.2)

        self.middle = conv_block(128, 256, dropout = 0.3)

        self.dec4 = conv_block(256, 128, dropout = 0.2)
        self.dec3 = conv_block(128 , 64, dropout = 0.2)
        self.dec2 = conv_block(64, 32, dropout = 0.1)
        self.dec1 = conv_block(32, 16, dropout = 0.1)

        self.upconv4 = upconv_block(256, 128)
        self.upconv3 = upconv_block(128, 64)
        self.upconv2 = upconv_block(64, 32)
        self.upconv1 = upconv_block(32, 16)

        self.cms_output = nn.ModuleList()
        for i in range(3):
            self.cms_output.append(lcm_layers(16))

        self.output = nn.Conv2d(16, 1, kernel_size = 1)

    def forward(self, x):

        self.output_cms = []

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

        for i in range(3):
            
            self.output_cms.append(self.cms_output[i](dec1))

        return self.output(dec1), self.output_cms
    
def initialize_model_lCM(img_channels):
    model = lCM_UNet(img_channels)
    return model