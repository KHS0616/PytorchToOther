import torch
import torch.nn as nn
import math

"""
LDSR 네트워크
에스프레소 미디어 응용개발 부서 정해서 주임연구원님이 개발한 자체 네트워크
Lighter, Deeper Super Resolution
"""
class LDSR(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=128, m=16):
        super(LDSR, self).__init__()
        act = nn.ReLU(True)
        n_feats = d
        n_shrinks = d//2
        n_blocks = m
        print(f'scale_factor : {scale_factor}')

        self.first_part = nn.Sequential(nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=5//2),nn.PReLU(n_feats))

        self.first_res = []
        for i in range(3):
            self.first_res .extend([ResBlock(
                    n_feats, 3, act=act, res_scale=1.0
                )])
        self.first_res = nn.Sequential(*self.first_res)

        self.mid_part =  nn.Sequential(nn.Conv2d(n_feats, n_shrinks, kernel_size=1), nn.PReLU(n_shrinks))

        self.mid_res = []
        for i in range(10):
            self.mid_res .extend([ResBlock(
                    n_shrinks, 3, act=act, res_scale=1.0
                )])
        self.mid_res = nn.Sequential(*self.mid_res)
        
        self.last_part =  nn.Sequential(nn.Conv2d(n_shrinks, n_feats, kernel_size=1), nn.PReLU(n_feats))

        self.last_res = []
        for i in range(3):
            self.last_res .extend([ResBlock(
                    n_feats, 3, act=act, res_scale=1.0
                )])
        self.last_res = nn.Sequential(*self.last_res)

        self.upsample = nn.Sequential(nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1), nn.PixelShuffle(scale_factor))

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.first_part(x)
        res = self.first_res(x)
        res += x
        x = self.mid_part(res)
        res = self.mid_res(x)
        res += x
        x = self.last_part(res)
        res = self.last_res(x)
        res += x
        x = self.upsample(res)
        x = x.permute(0,2,3,1)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, bias=bias, padding=3//2))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


####################################################


####################################################
class HSDSR_DENSE(nn.Module):
    def __init__(self, scale_factor=3, num_channels=3, main_blocks=14, sub_blocks=1, num_feats=256):
        super(HSDSR_DENSE, self).__init__()
        act = nn.ReLU(True)
        n_feats = num_feats
        n_shrinks = num_feats//2
        #self.first_part = nn.Sequential(nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=5//2),nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.first_part = nn.Sequential(nn.Conv2d(num_channels, n_feats, kernel_size=3, padding=3//2),nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.first_res = []
        for _ in range(sub_blocks):
            self.first_res .extend([DenseBlock_Bottleneck(
                    n_feats, res_scale=0.2
                )])
        self.first_res = nn.Sequential(*self.first_res)
        self.mid_part =  nn.Sequential(nn.Conv2d(n_feats, n_shrinks, kernel_size=1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.mid_res = []
        for _ in range(main_blocks):
            self.mid_res .extend([DenseBlock_Bottleneck(
                    n_shrinks, res_scale=0.2
                )])
        self.mid_res = nn.Sequential(*self.mid_res)
        self.last_part =  nn.Sequential(nn.Conv2d(n_shrinks, n_feats, kernel_size=1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.last_res = []
        for _ in range(sub_blocks):
            self.last_res .extend([DenseBlock_Bottleneck(
                    n_feats, res_scale=0.2
                )])
        self.last_res = nn.Sequential(*self.last_res)
        # self.upsample = nn.Sequential(nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=5, stride=1, padding=5//2), nn.PixelShuffle(scale_factor))
        self.upsample = nn.Sequential(nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1), nn.PixelShuffle(scale_factor))

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.first_part(x)
        res = self.first_res(x)
        res += x
        x = self.mid_part(res)
        res = self.mid_res(x)
        res += x
        x = self.last_part(res)
        res = self.last_res(x)
        res += x
        x = self.upsample(res)
        x = x.permute(0,2,3,1)
        return x


class DenseBlock_Bottleneck(nn.Module):
    def __init__(self, n_feats, res_scale=0.2):
        super(DenseBlock_Bottleneck, self).__init__()
        shink_feats = n_feats//4
        self.conv1 = nn.Sequential(nn.Conv2d(n_feats + 0 * shink_feats, shink_feats, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(n_feats + 1 * shink_feats, shink_feats, kernel_size=3, padding=1, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(n_feats + 2 * shink_feats, n_feats, kernel_size=1, bias=False), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.res_scale = res_scale
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        return conv3 * self.res_scale + x

####################################################
class HSDSR(nn.Module):
    def __init__(self, scale_factor, num_channels=3, d=128, m=16):
        super(HSDSR, self).__init__()
        act = nn.ReLU(True)
        n_feats = d
        n_shrinks = d//2
        n_blocks = m
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, n_feats, kernel_size=5, padding=5//2),
            nn.PReLU(n_feats)
        )
        self.mid_part = [nn.Conv2d(n_feats, n_shrinks, kernel_size=1), nn.PReLU(n_shrinks)]
        for i in range(n_blocks):
            self.mid_part .extend([HSDSR_ResBlock(
                    n_shrinks, res_scale=1.0
                )])
        self.mid_part.extend([nn.Conv2d(n_shrinks, n_feats, kernel_size=1), nn.PReLU(n_feats)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.Sequential(
            nn.Conv2d(n_feats, num_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
        )
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.first_part(x)
        res = self.mid_part(x)
        res += x
        x = self.last_part(res)
        x = x.permute(0,2,3,1)
        return x
class HSDSR_ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(HSDSR_ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3//2))
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

####################################################

import functools
import torch.nn.functional as F
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class BSRGAN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2):
        super(BSRGAN, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf
        print([in_nc, out_nc, nf, nb, gc, sf])

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out