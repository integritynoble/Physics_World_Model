"""HDNet original architecture for CASSI reconstruction.

From caiyuanhao1998/MST benchmark (CVPR 2022).
Reference PSNR: 34.97 dB on KAIST 256x256x28 (per-band, peak=1).
"""
import torch
import torch.nn as nn


def _default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class _OrigResBlock(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3):
        super().__init__()
        self.body = nn.Sequential(
            _default_conv(n_feats, n_feats, kernel_size, bias=True),
            nn.ReLU(True),
            _default_conv(n_feats, n_feats, kernel_size, bias=True),
        )

    def forward(self, x):
        return self.body(x) + x


class _DSC(nn.Module):
    def __init__(self, nin: int):
        super().__init__()
        self.conv_dws = nn.Conv2d(nin, nin, 1, 1, 0, groups=nin)
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_point = nn.Conv2d(nin, 1, 1, 1, 0, groups=1)
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.relu_dws(self.bn_dws(self.conv_dws(x)))
        out = self.maxpool(out)
        out = self.relu_point(self.bn_point(self.conv_point(out)))
        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1)).view(m, n, p, q)
        out = out.expand_as(x)
        return torch.mul(out, x) + x


class _EFF(nn.Module):
    def __init__(self, nin: int, nout: int, num_splits: int):
        super().__init__()
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [_DSC(nin // num_splits) for _ in range(num_splits)]
        )

    def forward(self, x):
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out = [self.subspaces[i](sub_feat[i]) for i in range(self.num_splits)]
        return torch.cat(out, dim=1)


class _SDL_attention(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.inter_planes = planes // 2
        self.conv_q_right = nn.Conv2d(inplanes, 1, 1, bias=False)
        self.conv_v_right = nn.Conv2d(inplanes, self.inter_planes, 1, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, planes, 1, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.conv_q_left = nn.Conv2d(inplanes, self.inter_planes, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(inplanes, self.inter_planes, 1, bias=False)
        self.softmax_left = nn.Softmax(dim=2)

    def spatial_attention(self, x):
        input_x = self.conv_v_right(x)
        b, c, h, w = input_x.size()
        input_x = input_x.view(b, c, h * w)
        context_mask = self.conv_q_right(x).view(b, 1, h * w)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = self.conv_up(context.unsqueeze(-1))
        return x * self.sigmoid(context)

    def spectral_attention(self, x):
        g_x = self.conv_q_left(x)
        b, c, h, w = g_x.size()
        avg_x = self.avg_pool(g_x).view(b, c, 1).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(b, self.inter_planes, h * w)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context).view(b, 1, h, w)
        return x * self.sigmoid(context)

    def forward(self, x):
        return self.spatial_attention(x) + self.spectral_attention(x)


class HDNetOriginal(nn.Module):
    """Original HDNet from caiyuanhao1998/MST benchmark.

    Architecture: head(Conv 28→64) → body(16 ResBlocks + SDL + EFF +
    15 ResBlocks + Conv) → tail(Conv 64→28). Input: 28-channel shift_back
    of measurement (no mask). Output: (bs, 28, H, W).
    """

    def __init__(self, in_ch=28, out_ch=28):
        super().__init__()
        n_feats = 64
        self.head = nn.Sequential(_default_conv(in_ch, n_feats, 3))
        m_body = [_OrigResBlock(n_feats) for _ in range(16)]
        m_body.append(_SDL_attention(n_feats, n_feats))
        m_body.append(_EFF(n_feats, n_feats, 4))
        m_body.extend([_OrigResBlock(n_feats) for _ in range(15)])
        m_body.append(_default_conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(_default_conv(n_feats, out_ch, 3))

    def forward(self, x, input_mask=None):
        x = self.head(x)
        res = self.body(x)
        res += x
        return self.tail(res)
