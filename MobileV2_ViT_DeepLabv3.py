
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViTBlock

class ASPP(nn.Module):
    
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_conv1x1_1 = nn.BatchNorm2d(out_channels)

        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv3x3_1 = nn.BatchNorm2d(out_channels)

        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv3x3_2 = nn.BatchNorm2d(out_channels)

        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv3x3_3 = nn.BatchNorm2d(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_conv1x1_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        feature_map_h = x.size()[2]
        feature_map_w = x.size()[3]

        out1 = F.relu(self.bn_conv1x1_1(self.conv1x1_1(x)))
        out2 = F.relu(self.bn_conv3x3_1(self.conv3x3_1(x)))
        out3 = F.relu(self.bn_conv3x3_2(self.conv3x3_2(x)))
        out4 = F.relu(self.bn_conv3x3_3(self.conv3x3_3(x)))

        out5 = self.avg_pool(x)
        out5 = F.relu(self.bn_conv1x1_2(self.conv1x1_2(out5)))
        out5 = F.interpolate(out5, size=(feature_map_h, feature_map_w), mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)

        return out
    

class DeepLabV3(nn.Module):
    def __init__(self, num_classes, backbone='mobilenetv2'):
        super(DeepLabV3, self).init()

        if backbone == 'mobilenetv2':
            self.backbone = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
            in_channels = 1280
        else:
            raise NotImplementedError

        self.aspp = ASPP(in_channels, out_channels=256)

        self.vit1 = ViTBlock(256, 256, 4, 4)
        self.vit2 = ViTBlock(256, 256, 4, 4)

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=48)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=48)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.Conv2d(in_channels=304, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)

def forward(self, x):
    x = self.backbone.features[:-1](x)
    x = self.aspp(x)
    x = self.vit1(x)
    x = self.vit2(x)
    x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
    x1 = self.conv1(x1)
    x1 = self.bn1(x1)
    x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
    x2 = self.conv2(x2)
    x2 = self.bn2(x2)
    x3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=True)
    x3 = self.conv3(x3)
    x3 = self.bn3(x3)
    x = torch.cat([x, x1, x2, x3], dim=1)
    x = self.conv4(x)
    x = self.bn4(x)
    x = self.conv5(x)

    return x
