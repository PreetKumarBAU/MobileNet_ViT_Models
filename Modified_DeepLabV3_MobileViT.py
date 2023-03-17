
import torch
import torch.nn as nn
import torch.nn.functional as F
#from vit_pytorch import ViTBlock
from einops import rearrange

from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, resnet50
from timm.models.vision_transformer import VisionTransformer

############################## For ResNet Blocks  ##############################

# Define ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x



############################## For Transformer Blocks  ##############################
## 1X1 Kernel
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

## 3x3 Kernel
def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


## Applying Norm before Block/Function
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



## Linear - SiLU - Linear 
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


## Attention Block
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)



## Transformer Architecture

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x



################################## MobileViTBlock  ########################################
class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = [ patch_size, patch_size]

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)        
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


'''
Use MobileNetV3 instead of MobileNetV2 as the backbone: MobileNetV3 is a more recent version of the MobileNet architecture that is more efficient and accurate than MobileNetV2. We can load it using torch.hub.load('pytorch/vision', 'mobilenet_v3_large', pretrained=True).

Use the DeepLabV3+ variant instead of the vanilla DeepLabV3. The DeepLabV3+ architecture includes an encoder-decoder pathway that helps to improve the segmentation performance. We can add the decoder part to our model by using the DeepLabV3Plus class from the torchvision.models.segmentation.deeplabv3.py module.

Use ViT-Large instead of ViT-Base: ViT-Large has more parameters and is capable of learning more complex representations than ViT-Base.

Increase the number of output channels in the ASPP module from 256 to 512: Increasing the number of output channels in the ASPP module can help to capture more fine-grained details in the input image.

Use GroupNorm instead of BatchNorm: GroupNorm has been shown to perform better than BatchNorm on small batch sizes and can help to improve the segmentation performance.
'''

############################## For ASPP Block  #######################################

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
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation

class DeepLabV3(nn.Module):
    def __init__(self, image_size , num_classes, backbone='mobilenetv3_large'):
        super(DeepLabV3, self).__init__()

        if backbone == 'mobilenetv3_large':
            ## Reduces by a factor of 32
            #self.backbone = mobilenet_v3_large(pretrained=True).features[:-1] 
            self.backbone = mobilenet_v3_large(pretrained=True).features[:-1]
            #in_channels = 960
            in_channels = 160
            
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            in_channels = 2048
        else:
            raise NotImplementedError
        
        self.img_size = image_size
        

        self.aspp = ASPP(in_channels, out_channels=256)

        #self.vit1 = VisionTransformer(img_size=self.img_size//32, patch_size=4, in_chans=256 * 5, num_classes=256 * 5, depth=4, num_heads=4)
        #self.vit2 = VisionTransformer(img_size=16, patch_size=4, in_chans=256 * 5, num_classes=256 * 5, depth=4, num_heads=4)

        #self.vit1 = ViTBlock(256, 256, 4, 4)
        #self.vit2 = ViTBlock(256, 256, 4, 4)
        self.kernel_size = 3
        self.patch_size = 2
        L = [2, 4, 3]     
        dims = [64, 80, 96]
        #dims = [96, 120, 144]
        self.MobileViT_1 = MobileViTBlock(dim = dims[0] , depth = L[0], channel = 256 * 5 , kernel_size = self.kernel_size, patch_size = self.patch_size, mlp_dim = int(dims[0]*2), dropout=0.)
        self.MobileViT_2 = MobileViTBlock(dim = dims[1] , depth = L[1], channel = 256  * 5 , kernel_size = self.kernel_size, patch_size = self.patch_size, mlp_dim = int(dims[1]*4), dropout=0.)

        self.conv1 = nn.Conv2d(in_channels=256 * 5, out_channels= 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 =  nn.GroupNorm(num_groups=32, num_channels= 256 )


        self.conv2 = nn.Conv2d(in_channels=256 * 5, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

        self.conv3 = nn.Conv2d(in_channels=256 * 5, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.GroupNorm(num_groups=32, num_channels=256)

        self.conv4 = nn.Conv2d(in_channels=256 * 8, out_channels=256 * 4 , kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.GroupNorm(num_groups=32 * 4, num_channels=256 * 4 )

        self.conv5 = nn.Conv2d(in_channels=256 * 4, out_channels=256  , kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.GroupNorm(num_groups=32 , num_channels=256 )
        '''
        self.up_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up_conv1 =nn.GroupNorm(num_groups=32 , num_channels=256 )
        self.relu_up_conv1 = nn.ReLU(inplace=True)

        self.up_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4, padding=1, bias=False)
        self.bn_up_conv2 =nn.GroupNorm(num_groups=32 , num_channels=256 )
        self.relu_up_conv2 = nn.ReLU(inplace=True)

        self.up_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4, padding=1, bias=False)
        self.bn_up_conv3 =nn.GroupNorm(num_groups=32 , num_channels=256 )
        self.relu_up_conv3 = nn.ReLU(inplace=True)
        '''
        out_channels = 256
        ## Decoder Block
        # Use the decoder module to recover the spatial information lost during the downsampling process
        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels , out_channels , kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels ),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels , out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )


        self.conv6 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x) ## As it is a LargeMobileV3 Arch ==> So it reduces the Size by a Large Value

        x = self.aspp(x)
        x = self.MobileViT_1(x)
        x = self.MobileViT_2(x)
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x3 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=True)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        
        ## Increase the Size of x1 , x2 , x3 to match the size of x
        x1 = F.interpolate(x1, scale_factor=1/0.5, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, scale_factor=1/0.25, mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, scale_factor=1/0.125, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1, x2,  x3], dim=1)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        '''
        x = self.up_conv1(x)
        x = self.bn_up_conv1(x)
        x = self.relu_up_conv1(x)

        x = self.up_conv2(x)
        x = self.bn_up_conv2(x)
        x = self.relu_up_conv2(x)

        x = self.up_conv3(x)
        x = self.bn_up_conv3(x)
        x = self.relu_up_conv3(x)
        '''
        x = self.decoder(x)
        x = self.conv6(x)
        return x
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    img = torch.randn(5, 3, 256, 256)  # 116633170
    print("Input Size::", img.shape)
    # MobileNetV2_UNet( vit_size, out_size=3, input_size=224, patch_size = 2, width_mult=1. )
    net = DeepLabV3(image_size = img.shape[2] ,  num_classes = 2)
    out = net(img)
    print(out.shape)
    print(count_parameters(net))

