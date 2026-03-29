import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

try:
    from torchvision.models import MobileNet_V2_Weights
    _HAS_ENUM_WEIGHTS = True
except Exception:
    _HAS_ENUM_WEIGHTS = False

# Import CBAM attention module
try:
    from utils.attention_modules import CBAM
    _HAS_CBAM = True
except ImportError:
    _HAS_CBAM = False

# Import SEBlock attention module
try:
    from utils.attention_modules import SEBlock
    _HAS_SE = True
except ImportError:
    _HAS_SE = False


# ---------- Basic Modules ----------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.GroupNorm(1, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class MiniASPP(nn.Module):
    """
    Minimal ASPP: 1x1 conv + multiple atrous convolutions + global pooling
    Reduced channels to 64 for extremely low GPU memory usage.
    """
    def __init__(self, in_ch, out_ch=64, aspp_rates=None):
        super().__init__()
        if aspp_rates is None:
            aspp_rates = [6]
        
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)
        )
        
        self.branches = nn.ModuleList()
        for rate in aspp_rates:
            branch = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        self.gp = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True)
        )
        
        total_branches = 2 + len(aspp_rates)
        self.out = nn.Sequential(
            nn.Conv2d(out_ch * total_branches, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.05)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        
        b0 = self.b0(x)
        branch_outputs = [b0]
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        gp = F.adaptive_avg_pool2d(x, 1)
        gp = self.gp(gp)
        gp = F.interpolate(gp, size=(h, w), mode="bilinear", align_corners=False)
        branch_outputs.append(gp)
        
        y = torch.cat(branch_outputs, dim=1)
        return self.out(y)


# ---------- MobileNetV2: 6-channel & OS=32 ----------
def _mobilenet_v2_6ch(pretrained=True, n_channels=6):
    if pretrained and _HAS_ENUM_WEIGHTS:
        backbone = tv.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    elif pretrained:
        backbone = tv.mobilenet_v2(pretrained=True)
    else:
        backbone = tv.mobilenet_v2(pretrained=False)

    first = backbone.features[0][0]
    new_first = nn.Conv2d(n_channels, first.out_channels, kernel_size=first.kernel_size,
                          stride=first.stride, padding=first.padding, bias=False)
    with torch.no_grad():
        if pretrained:
            w = first.weight
            if n_channels == 3:
                new_first.weight.copy_(w)
            elif n_channels > 3:
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = mean_w.repeat(1, n_channels, 1, 1)
                new_w[:, 0:3] = w
                new_first.weight.copy_(new_w + 0.01 * torch.randn_like(new_w))
            else:
                mean_w = w.mean(dim=1, keepdim=True)
                new_w = mean_w.repeat(1, n_channels, 1, 1)
                new_first.weight.copy_(new_w)
        else:
            nn.init.kaiming_normal_(new_first.weight, mode='fan_out', nonlinearity='relu')
    backbone.features[0][0] = new_first
    return backbone

def _set_output_stride(backbone, output_stride=32):
    """
    Configure output stride for MobileNetV2:
    - output_stride=8: modify stride and dilation in features[7] and features[14]
    - output_stride=16: modify stride and dilation in features[14]
    - output_stride=32: keep default settings
    """
    if output_stride == 32:
        return backbone
    
    if output_stride == 16:
        idx = 14
        module = backbone.features[idx]
        conv_layer = module.conv[1][0]
        
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=(1, 1),
            padding=(2, 2),
            dilation=(2, 2),
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None
        )
        
        with torch.no_grad():
            new_conv.weight.copy_(conv_layer.weight)
            if conv_layer.bias is not None:
                new_conv.bias.data.copy_(conv_layer.bias.data)
        
        module.conv[1][0] = new_conv
        module.stride = 1
    
    elif output_stride == 8:
        for idx, dilation in [(7, 2), (14, 4)]:
            module = backbone.features[idx]
            conv_layer = module.conv[1][0]
            
            new_conv = nn.Conv2d(
                conv_layer.in_channels,
                conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=(1, 1),
                padding=(dilation, dilation),
                dilation=(dilation, dilation),
                groups=conv_layer.groups,
                bias=conv_layer.bias is not None
            )
            
            with torch.no_grad():
                new_conv.weight.copy_(conv_layer.weight)
                if conv_layer.bias is not None:
                    new_conv.bias.data.copy_(conv_layer.bias.data)
            
            module.conv[1][0] = new_conv
            module.stride = 1
    
    return backbone


# ---------- Ultra-Light DeepLabV3+ ----------
class UltraLightDeepLabV3Plus(nn.Module):
    """
    Ultra-lightweight version for low memory usage:
      - OS=32 (backbone output 1/32 scale)
      - ASPP channels = 64
      - Lightweight decoder with depthwise conv
      - Low-level skip connection from 1/4 scale
      - Optional CBAM / SE attention modules
    """
    def __init__(self, n_channels=6, n_classes=1,
                 pretrained_backbone=True, aspp_out=64, dec_ch=64, low_ch_out=32, low_ch_in=32,
                 use_cbam=False, cbam_reduction_ratio=16, output_stride=32, aspp_rates=None, class_prior=None, use_se=False):
        super().__init__()

        self.backbone = _mobilenet_v2_6ch(pretrained=pretrained_backbone, n_channels=n_channels)
        _set_output_stride(self.backbone, output_stride)

        self.low_slice_end = 4
        self.high_slice_start = 5

        self.aspp = MiniASPP(in_ch=1280, out_ch=aspp_out, aspp_rates=aspp_rates)

        self.decoder = DepthwiseSeparableConv(aspp_out + low_ch_out, dec_ch)
        self.classifier = nn.Conv2d(dec_ch, n_classes, 1)

        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch_in, low_ch_out, 1, bias=False),
            nn.GroupNorm(1, low_ch_out),
            nn.ReLU(inplace=True)
        )
        
        self.low_reduce = self.low_proj
        
        self.use_cbam = use_cbam and _HAS_CBAM
        if self.use_cbam:
            self.cbam_after_aspp = CBAM(aspp_out, reduction_ratio=cbam_reduction_ratio)
            self.cbam_after_decoder = CBAM(dec_ch, reduction_ratio=cbam_reduction_ratio)
        
        self.use_se = use_se and _HAS_SE
        if self.use_se:
            self.se_after_aspp = SEBlock(aspp_out, reduction_ratio=16)
            self.se_after_decoder = SEBlock(dec_ch, reduction_ratio=16)
        
        self.class_prior = class_prior
        
        self._init_head()
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _forward_slices(self, x):
        feats = self.backbone.features
        low = x
        for i in range(0, self.low_slice_end + 1):
            low = feats[i](low)
        high = low
        for i in range(self.high_slice_start, len(feats)):
            high = feats[i](high)
        return low, high

    def forward(self, x):
        b, c, h, w = x.shape
        low, high = self._forward_slices(x)

        aspp = self.aspp(high)
        aspp = F.interpolate(aspp, size=low.shape[-2:], mode="bilinear", align_corners=False)
        
        if self.use_cbam:
            aspp = self.cbam_after_aspp(aspp)
            
        if self.use_se:
            aspp = self.se_after_aspp(aspp)
            
        low = self.low_reduce(low)
        y = torch.cat([aspp, low], dim=1)
        y = self.decoder(y)
        
        if self.use_cbam:
            y = self.cbam_after_decoder(y)
            
        if self.use_se:
            y = self.se_after_decoder(y)
            
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        y = self.classifier(y)
        return y

    def _init_head(self):
        for m in [self.aspp, self.decoder, self.classifier]:
            for mod in m.modules() if isinstance(m, nn.Module) else []:
                if isinstance(mod, nn.Conv2d):
                    nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
                elif isinstance(mod, nn.GroupNorm):
                    nn.init.ones_(mod.weight)
                    nn.init.zeros_(mod.bias)


def get_ultra_light_deeplabv3_plus(n_channels=6, n_classes=1,
                                   pretrained_backbone=True,
                                   aspp_out=64, dec_ch=64, low_ch_out=32,
                                   use_cbam=False, cbam_reduction_ratio=16,
                                   output_stride=32, aspp_rates=None, class_prior=None, use_se=False):
    return UltraLightDeepLabV3Plus(
        n_channels=n_channels, n_classes=n_classes,
        pretrained_backbone=pretrained_backbone,
        aspp_out=aspp_out, dec_ch=dec_ch, low_ch_out=low_ch_out,
        use_cbam=use_cbam, cbam_reduction_ratio=cbam_reduction_ratio,
        output_stride=output_stride, aspp_rates=aspp_rates, 
        class_prior=class_prior, use_se=use_se
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_ultra_light_deeplabv3_plus(pretrained_backbone=True)
    x = torch.randn(1, 6, 256, 256)
    y = model.train()(x)
    print("input:", x.shape, "output:", y.shape)
    print("params:", f"{count_parameters(model):,}")
