# MS-RDDNet
This is the official code implementation of MS-RDDNet for the *Neurocomputing* journal paper: **"Revisiting Multi-Scale Feature Representation and Fusion for UAV-Based Road Distress Detection"**

<div align="center">
    <img src="Fig. 1-Network structure of MS-RDDNet.png" width="600">
</div>

<div align="center">​ 
Network structure of MS-RDDNet
</div>	

## Datasets
```python
Yan H, Zhang J. UAV-PDD2023: A benchmark dataset for pavement distress detection based on UAV images[J]. Data in Brief, 2023, 51: 109692.

He J, Gong L, Xu C, et al. HighRPD: A high-altitude drone dataset of road pavement distress[J]. Data in Brief, 2025, 59: 111377.



```



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

```python
class MultiScaleAttnConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dws = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, 1)
            )
            for k in [3, 5, 7]
        ])
        self.fusion = nn.Conv2d(in_channels*3, in_channels, 1)
        # attention weight generator
        self.attn = nn.Conv2d(in_channels, 3, 1)  

    def forward(self, x):
        feats = [dw(x) for dw in self.dws]
        fused = torch.cat(feats, dim=1)
        fused = self.fusion(fused)
        
        # Softmax-normalized attention weights
        attn_weights = F.softmax(self.attn(x), dim=1)
        # weighted fusion
        out = torch.sum(attn_weights.unsqueeze(2) * fused.unsqueeze(1), dim=1)
        return out  


class Detect(nn.Module):
    dynamic = False  
    export = False 
    format = None  
    end2end = False  
    max_det = 300  
    shape = None
    anchors = torch.empty(0)  
    strides = torch.empty(0)  
    legacy = False  

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  
        self.nl = len(ch)  
        self.reg_max = 16  
        
        self.no = nc + self.reg_max * 4  
        # strides computed during build
        self.stride = torch.zeros(self.nl)  
        c2, c3 = max((16, ch[0]//4, self.reg_max*4)), max(ch[0], min(self.nc, 100))  
        
        # bounding box prediction branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                MultiScaleAttnConv(x),  # multi-scale attention convolution
                DWConv(x, c2, 3), 
                DWConv(c2, c2, 3), 
                nn.Conv2d(c2, 4*self.reg_max, 1)
            ) for x in ch
        )
        
        # class prediction branch
        self.cv3 = (nn.ModuleList(
            nn.Sequential(
                MultiScaleAttnConv(x),  # multi-scale attention convolution
                DWConv(x, c3, 3), 
                DWConv(c3, c3, 3), 
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch)
                    
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    MultiScaleAttnConv(x),  # 引入多尺度注意力卷积
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        if self.end2end:
            return self.forward_end2end(x)
        
        for i in range(self.nl):
            # (bs, 4*self.reg_max+nc, 80, 80)
            # (bs, 4*self.reg_max+nc, 40, 40)
            # (bs, 4*self.reg_max+nc, 20, 20)
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training: 
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
```
