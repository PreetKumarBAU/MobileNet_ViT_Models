
## Changes needed are : Use Concatenation instead of Addition of FramesFeatures , FlowFeatures and FirstFrame
## Use More Frames say 4 or 5 or 6 Frames

import torch
import torch.nn as nn
from einops import rearrange

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



## MobileV2 Block ==> 1x1 CNN - 3x3 DW CNN - 1x1 CNN
## [ from in_ch to in_ch*expansion-  in_ch*expansion to in_ch*expansion  - in_ch*expansion to in_ch  ]
## With Residual Connection if   ==> self.use_res_connect = self.stride == 1 and inp == oup

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.in1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.in2 = nn.BatchNorm2d(out_c)
        self.silu = nn.SiLU()




    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.in1(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = self.silu(x)

        return x


class MV2BlockTransposeConv(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
        
            self.conv = nn.Sequential(

                # dw
                
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 2 , stride, 0, groups=hidden_dim, bias=False) if self.stride == 2 else nn.Conv2d(hidden_dim, hidden_dim, 3 , stride, 1, groups=hidden_dim, bias=False),
                
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),

                ## ConvTranspose2d
                # dw  
                nn.ConvTranspose2d(hidden_dim, hidden_dim, 2 , stride, 0, groups=hidden_dim, bias=False) if self.stride == 2 else nn.Conv2d(hidden_dim, hidden_dim, 3 , stride, 1, groups=hidden_dim, bias=False),
                #nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



## MobileVIT Block ==> 3x3 CNN - 1x1 CNN -  TransformerBlock - 1x1 CNN - 3x3 CNN
## Rearrange before and after Transformer Block
## rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
## rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

## out_ch = in_ch

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

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



## MobileVIT Classifier Architecture
class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x




def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


## Take 3 Frames in Encoder , ## Take 2 Flow Frames in Encoder,  ## Take 1 Frames in Encoder, Outputs a Single Mask for now

## MobileVIT Segmentation Architecture
class MobileViTSegmentation(nn.Module):
    def __init__(self, image_size, dims, channels, out_ch= 3 , expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

###################  3 Frames  ########################
        ## Encoder Part For 3 Frames 
        self.initial_cnn_frames = conv_nxn_bn(9, channels[0], stride=1)
        self.conv1_frames = conv_nxn_bn(channels[0], channels[0], stride=2)

        self.mv2_frames = nn.ModuleList([])
        self.mv2_frames.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2_frames.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2_frames.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2_frames.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2_frames.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2_frames.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2_frames.append(MV2Block(channels[7], channels[8], 2, expansion))

        ## MobileViTBlock for 3 Frames
        self.mvit_frames = nn.ModuleList([])
        self.mvit_frames.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit_frames.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))

        
        self.three_frames_encoder = nn.Sequential(
            self.initial_cnn_frames, 
            self.conv1_frames,
            *self.mv2_frames[:5],
            self.mvit_frames[0],
            self.mv2_frames[5] ,
            self.mvit_frames[1],
            self.mv2_frames[6]

        )
###################  2 Flows  ########################

        ## Encoder Part For 2 Flow Frames 
        self.initial_cnn_flows = conv_nxn_bn(6, channels[0], stride=1)
        self.conv1_flows = conv_nxn_bn(channels[0], channels[0], stride=2)
        self.mv2_flows = nn.ModuleList([])
        self.mv2_flows.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2_flows.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2_flows.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2_flows.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2_flows.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2_flows.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2_flows.append(MV2Block(channels[7], channels[8], 2, expansion))

        ## MobileViTBlock for 2 Flows
        self.mvit_flows = nn.ModuleList([])
        self.mvit_flows.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit_flows.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))

        self.two_flows_encoder = nn.Sequential(
            self.initial_cnn_flows, 
            self.conv1_flows,
            *self.mv2_flows[:5],
            self.mvit_flows[0],
            self.mv2_flows[5] ,
            self.mvit_flows[1],
            self.mv2_flows[6]

        )

###################  1 Frame  ########################
        ## Encoder Part for 1 Frame
        ## Initial CNN layer
        self.initial_cnn = conv_nxn_bn(3, channels[0], stride=1)
        self.conv1 = conv_nxn_bn(channels[0], channels[0], stride=2)

        ## MobileNet Blocks
        self.mv2 = nn.ModuleList([])
        self.mv2_convtranspose =  nn.ModuleList([])
        self.conv_blocks =  nn.ModuleList([])

        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
    ### BottleNeck
        ## MobileViTBlock 
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

    ## Decoder Block
        ## MobileViTBlock for Decoder
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))

        ## MobileNet Blocks with Conv2dTranspose
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[8], channels[7], 2, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[6], channels[5], 2, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[4], channels[3], 2, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[3], channels[2], 1, expansion))   # Repeat
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[3], channels[2], 1, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[2], channels[1], 2, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[1], channels[0], 1, expansion))
        self.mv2_convtranspose.append(MV2BlockTransposeConv(channels[1], channels[0], 2, expansion))

        self.output_cnn = conv_1x1_bn(channels[0], out_ch)

        '''
        ## conv_blocks for Reduces the out_ch + skip_ch to out_ch ==> Used after Skip Features Concatenation
        self.conv_blocks.append( conv_block(channels[7] + skip_ch1 , channels[7]))
        self.conv_blocks.append( conv_block(in_c, out_c))
        self.conv_blocks.append( conv_block(in_c, out_c))
        self.conv_blocks.append( conv_block(in_c, out_c))
        '''

        self.pool = nn.AvgPool2d(ih//32, 1)
        self.fc = nn.Linear(channels[-1], out_ch, bias=False)


## Shortcut is before the "Reduction"
    def forward(self, frames , flows ):

        First_Frame = frames[0]
        conc_frames = torch.concat(frames , dim = 1)
        conc_flows = torch.concat(flows , dim = 1)

        flows_features  = self.two_flows_encoder(conc_flows)
        frames_features = self.three_frames_encoder(conc_frames)

        shortcut = []
        x = self.initial_cnn(First_Frame)
        shortcut.append([x, x.shape[1]])
        x = self.conv1(x)       ## Reduces the Image by 2 using Strides
        x = self.mv2[0](x)
        shortcut.append([x, x.shape[1]])
        x = self.mv2[1](x)      ## Reduces the Image by 2 using Strides
        x = self.mv2[2](x)      
        x = self.mv2[3](x)      # Repeat
        shortcut.append([x, x.shape[1]])
        x = self.mv2[4](x)      ## Reduces the Image by 2 using Strides
        x = self.mvit[0](x)

        shortcut.append([x, x.shape[1]])
        x = self.mv2[5](x)      ## Reduces the Image by 2 using Strides
        x = self.mvit[1](x)

        shortcut.append([x,x.shape[1]])
        x = self.mv2[6](x)      ## Reduces the Image by 2 using Strides

        ### Add the "Frames_Features" with "Flow_Features" with "OneFrame_Features"
        x = frames_features + flows_features + x


        x = self.mvit[2](x)
        #x = self.conv2(x)


        ## Decoder Block
        x = self.mvit[3](x)
        x= self.mv2_convtranspose[0](x)  ## Increases the Image by 2 using Conv2dTranspose
        out_ch = x.shape[1]
        
        
        ## Conconate the Last Shortcut Connection/Feature with x and pop it out
        x1 = torch.concat([x, shortcut[-1][0]], dim=1)
        

        ## To reduce the Channels from out_ch + skip_ch to out_ch
        x = conv_block(out_ch + shortcut[-1][1] , out_ch)(x1)
        shortcut.pop(-1)

        x = self.mvit[4](x)
        x = self.mv2_convtranspose[1](x)  ## Increases the Image by 2 using Conv2dTranspose
        out_ch = x.shape[1]

        ## Conconate the Last Shortcut Connection/Feature with x and pop it out
        x2 = torch.concat([x, shortcut[-1][0]], dim = 1)
        
        ## To reduce the Channels from out_ch + skip_ch to out_ch
        x = conv_block(out_ch + shortcut[-1][1] , out_ch)(x2)
        shortcut.pop(-1)

        x = self.mvit[5](x)
        x = self.mv2_convtranspose[2](x)  ## Increases the Image by 2 using Conv2dTranspose
        out_ch = x.shape[1]

        ## Conconate the Last Shortcut Connection/Feature with x and pop it out
        x3 = torch.concat([x, shortcut[-1][0]] , dim = 1)
        
        ## To reduce the Channels from out_ch + skip_ch to out_ch
        x = conv_block(out_ch + shortcut[-1][1] , out_ch)(x3)
        shortcut.pop(-1)

        x = self.mv2_convtranspose[3](x)
        x = self.mv2_convtranspose[4](x)
        x = self.mv2_convtranspose[5](x)
        out_ch = x.shape[1]

        ## Conconate the Last Shortcut Connection/Feature with x and pop it out
        x4 = torch.concat([x, shortcut[-1][0]], dim = 1)
        
        ## To reduce the Channels from out_ch + skip_ch to out_ch
        x = conv_block(out_ch + shortcut[-1][1] , out_ch)(x4)
        shortcut.pop(-1)

        x = self.mv2_convtranspose[6](x)
        x = self.mv2_convtranspose[7](x)
        out_ch = x.shape[1]

        ## Conconate the Last Shortcut Connection/Feature with x and pop it out
        x5 = torch.concat([x, shortcut[-1][0]], dim = 1)
        
        ## To reduce the Channels from out_ch + skip_ch to out_ch
        x = conv_block(out_ch + shortcut[-1][1] , out_ch)(x5)
        shortcut.pop(-1)

        x = self.output_cnn(x)

        #x = self.pool(x).view(-1, x.shape[1])
        #x = self.fc(x)
        return x



def mobilevit_seg_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViTSegmentation((256, 256), dims, channels, out_ch=3, expansion=2)


def mobilevit_seg_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViTSegmentation((256, 256), dims, channels, out_ch=3)



def mobilevit_seg_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViTSegmentation((256, 256), dims, channels, out_ch=3)


if __name__ == '__main__':
    img1 = torch.randn(5, 3, 256, 256)
    img2 = torch.randn(5, 3, 256, 256)
    img3 = torch.randn(5, 3, 256, 256)

    flow1 = torch.randn(5, 3, 256, 256)
    flow2 = torch.randn(5, 3, 256, 256)
    frames = [img1 , img2 ,  img3 ]
    flows = [flow1 , flow2 ]

    vit = mobilevit_seg_xxs()
    out = vit(frames , flows)
    print(out.shape)
    print(count_parameters(vit))

    '''
    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
    
    
    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
    '''

