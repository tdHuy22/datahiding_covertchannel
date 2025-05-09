import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedInceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, dilation=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=2, dilation=2),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=3, dilation=3),
        )
        self.conv_cat = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.conv_cat(out)
        return out

class SecretEmbedder(nn.Module):
    def __init__(self, concat_channels = 6):
        super(SecretEmbedder, self).__init__()
        self.concat_channels = concat_channels
    
    def forward(self, x1, x2):
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)
        if x.size(1) != self.concat_channels:
            raise ValueError(f"Expected {self.concat_channels} channels, but got {x.size(1)}")
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels = 6, base_channels = 64):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dilated2 = DilatedInceptionModule(base_channels, base_channels * 2)        # 64 -> 128
        self.dilated3 = DilatedInceptionModule(base_channels * 2, base_channels * 4)    # 128 -> 256
        self.dilated4 = DilatedInceptionModule(base_channels * 4, base_channels * 8)    # 256 -> 512
        self.dilated5 = DilatedInceptionModule(base_channels * 8, base_channels * 8)    # 512 -> 512
    
    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.dilated2(enc1)
        enc3 = self.dilated3(enc2)
        enc4 = self.dilated4(enc3)
        enc5 = self.dilated5(enc4)
        return enc1, enc2, enc3, enc4, enc5

class StegoReconstructor(nn.Module):
    def __init__(self, base_channels = 64, out_channels = 3):
        super(StegoReconstructor, self).__init__()
        self.dilated6 = DilatedInceptionModule(base_channels * 8, base_channels * 8)   # 512 -> 512
        self.dilated7 = DilatedInceptionModule(base_channels * 16, base_channels * 4)  # 1024 -> 256
        self.dilated8 = DilatedInceptionModule(base_channels * 8, base_channels * 2)   # 512 -> 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Đầu ra: N × N × 3 (stego)
        )
    
    def forward(self, enc1, enc2, enc3, enc4, enc5):
        dec5 = self.dilated6(enc5)
        dec4 = self.dilated7(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dilated8(torch.cat([dec4, enc3], dim=1))
        dec2 = self.conv2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.conv_out(torch.cat([dec2, enc1], dim=1))
        return dec1

class SecretExtractor(nn.Module):
    def __init__(self, in_channels = 3, base_channels = 64, out_channels = 3):
        super(SecretExtractor, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # Đầu ra: N × N × 3 (secret)
        )
    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv_out(x)
        return x

class SteganoModel(nn.Module):
    def __init__(self, in_channels = 6, base_channels = 64, out_channels = 3):
        super(SteganoModel, self).__init__()
        self.secret_embedder = SecretEmbedder(concat_channels=in_channels)
        self.feature_extractor = FeatureExtractor(in_channels=in_channels, base_channels=base_channels)
        self.stego_reconstructor = StegoReconstructor(base_channels=base_channels, out_channels=out_channels)
        self.secret_extractor = SecretExtractor(in_channels=in_channels // 2, base_channels=base_channels, out_channels=out_channels)


    def forward(self, cover_image, secret_image):
        # Embed secret into cover image
        embedded = self.secret_embedder(cover_image, secret_image)
        
        # Extract features from embedded image
        enc1, enc2, enc3, enc4, enc5 = self.feature_extractor(embedded)
        
        # Reconstruct stego image
        stego_image = self.stego_reconstructor(enc1, enc2, enc3, enc4, enc5)
        
        # Extract secret from stego image
        extracted_secret = self.secret_extractor(stego_image)
        
        return stego_image, extracted_secret
        