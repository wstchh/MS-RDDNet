# MS-RDDNet
This is the official code implementation of MS-RDDNet for the *Neurocomputing* journal paper: **"Revisiting Multi-Scale Feature Representation and Fusion for UAV-Based Road Distress Detection"**

<div align="center">
    <img src="Fig. 1-Network structure of MS-RDDNet.png" width="600">
</div>

<div align="center">​ 
Network structure of MS-RDDNet
</div>	

## 1. MSGC module

<div align="center">
    <img src="Fig. 2-The structure of MSGC module.png" width="600">
</div>

<div align="center">​ 
The structure of MSGC module
</div>	


```python
class GatedFusion(nn.Module):
    def __init__(self, num_scales, in_channels):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels*num_scales, num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.out_conv = Conv(in_channels, in_channels, k=1)

    def forward(self, x_list):
        # [B, n*C, H, W]
        feats = torch.cat(x_list, dim=1)  
        # [B, n, 1, 1]
        gates = self.gate_gen(feats)

        b, c, h, w = x_list[0].shape
        gates = gates.view(b, -1, 1, 1, 1).expand(-1, -1, c, h, w).contiguous()
        # [B, n, C, H, W]
        feats = torch.stack(x_list, dim=1)  
        # [B, C, H, W]
        fused = torch.sum(gates * feats, dim=1) 
        return self.out_conv(fused)


class MSGC(nn.Module):
    "Multi-Scale Gated Convolution"
    def __init__(self, c1, c2, shortcut=True, k=(3,5,7,9), e=0.5):  
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)

        # Multi-Scale Large Kernel Depthwise Separable Convolutions
        self.large_kernel_convs = nn.ModuleList()
        for kernel in k:
            padding = kernel // 2  
            self.large_kernel_convs.append(
                nn.Sequential(
                    nn.Conv2d(c_, c_, kernel_size=kernel, stride=1, padding=padding, groups=c_, bias=False),
                    nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )

        self.fuser = GatedFusion(len(k), c2)
        
        self.cv2 = Conv(len(k) * c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        features = []

        for conv in self.large_kernel_convs:
            features.append(conv(x))
        y = self.fuser(features)

        if self.add:
            y = y + identity
        return y
```


<div align="center">
    <img src="Fig. 3-The structure of C2f-MSGC module.png" width="500">
</div>

<div align="center">​ 
The structure of C2f-MSGC module
</div>	

```python
class C2f_MSGC(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n)*self.c, c2, 1)  
        self.m = nn.Sequential(*(MSGC(self.c, self.c) for _ in range(n)))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

```



## 2. MSDHA mechanism

<div align="center">
    <img src="Fig. 4-The structure of MSDHA  mechanism.png" width="600">
</div>

<div align="center">​ 
The structure of MSDHA mechanism
</div>

```python
class CSAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CSAttention, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        y_channel = self.avg_pool(x) + self.max_pool(x)
        y_channel = self.channel_conv(y_channel)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y_spatial = torch.cat([avg_out, max_out], dim=1)
        y_spatial = self.spatial_conv(y_spatial)

        y = y_channel * y_spatial
        return x * y.expand_as(x)


class MSDHA(nn.Module):
    def __init__(self, c1, factor=4, reduction=16):
        super(MSDHA, self).__init__()
        c_ = int(c1 // factor)
        
        # 1x1 convolution for channel compression
        self.conv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0)

        # 3×3 convolution with dilation=2, receptive field 5×5
        self.conv3x3_d2 = nn.Sequential(
                            nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=2, dilation=2, groups=c_, bias=False),
                            nn.ReLU(inplace=True)
                        )
                                    
        # 3×3 convolution with dilation=4, receptive field 9×9
        self.conv3x3_d4 = nn.Sequential(
                            nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=4, dilation=4, groups=c_, bias=False),
                            nn.ReLU(inplace=True)
                        )

        # 3×3 convolution with dilation=6, receptive field 13×13
        self.conv3x3_d6 = nn.Sequential(
                            nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=6, dilation=6, groups=c_, bias=False),
                            nn.ReLU(inplace=True)
                        )

        # 1×1 convolution for channel adjustment
        self.conv2 = nn.Conv2d(c_*4, c_*4, kernel_size=1, stride=1, padding=0)

        self.csa = CSAttention(c_*4, reduction)
       

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv3x3_d2(x)
        x4 = self.conv3x3_d4(x)
        x6 = self.conv3x3_d6(x)
        x_concat = torch.cat([x, x2, x4, x6], dim=1)
        out = self.conv2(x_concat)

        # residual attention enhancement
        out = self.csa(out)
        return out
```

## 3. LMSADet head

<div align="center">
    <img src="Fig. 5-The structure of LMSADet head.png" width="700">
</div>

<div align="center">​ 
The structure of LMSADet head
</div>

