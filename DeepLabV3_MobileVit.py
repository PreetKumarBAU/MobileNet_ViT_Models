import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from einops.layers.torch import Rearrange

# In this code, we first load the MobileNetv3 and ViT models using `torchvision.models` and `torch.hub.load`, respectively. We then replace the last few layers of MobileNetv3 with the ASPP module and use the decoder module to recover the spatial information lost during the downsampling process. Finally, we copy the convolutional and transformer layers from MobileNetv3 and ViT into our model, respectively. In the forward pass, we first pass the input image through the encoder, then through the transformer, and finally through the decoder.


'''
In the forward method, we first pass the input image through the encoder, which consists 
of the MobileNetv3 layers. We then pass the output of the encoder through the ViT blocks 
to obtain a high-level representation of the input. We then pass this high-level representation
through the ASPP module to obtain a dense prediction map. Finally, we use the decoder module
to recover the spatial information lost during the downsampling process and obtain the final
prediction map.

Note that the number of ViT blocks can be adjusted by setting the num_blocks parameter
when creating an instance of the DeepLabV3_MobileVit class. The default value of num_blocks
is 12, but it can be increased or decreased depending on the complexity of the input images
and the available computational resources.

'''


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
    
class DeepLabV3_MobileVit(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_MobileVit, self).__init__()

        # Load MobileNetv3 model
        mobilenet = models.mobilenet_v3_large(pretrained=True)

        # Load ViT model
        vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        # Replace the last few layers of the MobileNetv3 model with the ASPP module
        in_channels = 960  # output channels of the last convolutional layer in MobileNetv3
        out_channels = 256
    
        ## ASSP Block
        self.aspp1 = ASPP(in_channels, out_channels=256) ## 

        self.aspp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False),
        )

        ## Decoder Block
        # Use the decoder module to recover the spatial information lost during the downsampling process
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, bias=False),
        )

        # Copy the convolutional layers from MobileNetv3 into the model
        self.conv1 = mobilenet.features[0:3]
        self.conv2 = mobilenet.features[3:5]
        self.conv3 = mobilenet.features[5:11]
        self.conv4 = mobilenet.features[11:16]
        self.conv5 = mobilenet.features[16:17]

        # Copy the transformer layers from ViT into the model
        self.blocks = nn.Sequential(*list(vit.blocks.children()))

        # Remove the classification head from the ViT model
        vit.blocks = vit.blocks[:-1]

        # Reshape the output of the MobileNetv3 model to match the expected input shape of the ViT model
        # The input tensor shape should be [batch_size, sequence_length, hidden_size]
        # We want to reshape the tensor to [batch_size, hidden_size, sequence_length]
        mobilenet_output_shape = (1, -1, 960)
        vit_input_shape = (768, -1)

        self.reshape = nn.Sequential(
                    Rearrange(pattern='b c h w -> b (h w) c', h=7, w=7),
                    nn.Linear(in_features=960, out_features=768),
                    nn.ReLU(inplace=True),
                    nn.Linear(in_features=768, out_features=768),
                    nn.ReLU(inplace=True),
                    vit.blocks,
                    #nn.LayerNorm(768),
                    nn.Linear(in_features=768, out_features=960),
                    nn.ReLU(inplace=True),
                    Rearrange(pattern='b (h w) c -> b c h w', h=7, w=7),
                )

        # Combine the MobileNetv3 and ViT models
        self.features1 = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.reshape,
            self.aspp1,
            self.decoder,
        )
        # Add Upsample layer to increase the size of the output tensor
        #self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features1(x)
        #x = self.upsample(x)
        return x

'''
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Transformer
        x = self.blocks(x)

        # Decoder
        x = self.aspp(x)
        x = F.interpolate(x, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat([x, self.conv4[1].running_mean], dim=1))
        x = F.interpolate(x, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bilinear', align_corners=False)

        return x
    
'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
if __name__ == '__main__':
    img = torch.randn(5, 3, 224, 224)   # 104029168
    print("Input Size::", img.shape)
    # MobileNetV2_UNet( vit_size, out_size=3, input_size=224, patch_size = 2, width_mult=1. )
    net = DeepLabV3_MobileVit(  num_classes = 2)
    out = net(img)
    print(out.shape)
    print(count_parameters(net))