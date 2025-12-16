"""
模型定义文件

Author: 罗涛
Date: 2024-10-12
"""


import time
from torch import nn
import torch
from einops import rearrange



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class Block(nn.Module):
    def __init__(self, n_layer, n_feat, num_heads, window_size, mlp_ratio):
        super(Block, self).__init__()
        # self.layers = nn.ModuleList([BasicLayer(n_feat,
        #                                         num_heads,
        #                                         window_size,
        #                                         mlp_ratio)
        #                              for i in range(n_layer)])

        self.layers = nn.ModuleList([SHSA_B(n_feat)
                                     for i in range(n_layer)])

        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x):
        shotcut = x
        for layer in self.layers:
            x = layer(x)
        x = self.conv(x)
        return x + shotcut


class MLP(nn.Module):

    def __init__(self, in_feat, h_feat=None, out_feat=None):
        super().__init__()

        self.fc1 = nn.Conv2d(in_channels=in_feat, out_channels=h_feat, kernel_size=1, padding=0, stride=1, groups=1,
                             bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels=h_feat, out_channels=out_feat, kernel_size=1, padding=0, stride=1, groups=1,
                             bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class CAB(nn.Module):
    def __init__(self, num_features, reduction=4):
        super(CAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_features // 4, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return self.module(x)


class WindowAttention(nn.Module):
    def __init__(self, n_feat, num_heads, window_size, bias=True):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        # self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Linear(n_feat, n_feat * 3, bias=bias)
        self.proj_out = nn.Linear(n_feat, n_feat)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (B, C, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, C, H, W)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

    def window_attention(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, heads, N, C//heads]
        attn = (q @ k.transpose(-2, -1).contiguous()) * (self.num_heads ** -0.5)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj_out(x)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x_windows = self.window_partition(x, self.window_size)  # [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]

        attn_windows = self.window_attention(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        out = self.window_reverse(attn_windows, self.window_size, H, W)  # [B, C, H, W]
        return out


class MHSAttention(nn.Module):
    def __init__(self, n_feat, num_heads, bias=True):
        super(MHSAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(n_feat, n_feat * 3, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def global_attention(self, q, k, v):
        B, C, H, W = q.shape
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        attn = (q @ k.transpose(-2, -1).contiguous()) * (self.num_heads ** -0.5)  # [b head l l]
        attn = attn.softmax(dim=-1)  # [b head l l]
        out = (attn @ v)  # [b head l c]
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', h=H, w=W)
        return out

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        out = self.global_attention(q, k, v)

        out = self.project_out(out)

        return out


class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
            self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim
        self.pre_norm = GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SHSA_B(nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class BasicLayer(nn.Module):
    def __init__(self, n_feat, num_heads, window_size, mlp_ratio):
        super(BasicLayer, self).__init__()

        self.norm_ra = LayerNorm2d(n_feat)
        self.ra = MHSAttention(n_feat, num_heads)
        self.lambda_ra = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

        self.norm_ra_mlp = LayerNorm2d(n_feat)
        self.mlp_ra = MLP(n_feat, int(n_feat * mlp_ratio), n_feat)

        self.norm_wa = LayerNorm2d(n_feat)
        self.wa = WindowAttention(n_feat, num_heads, window_size)
        self.lambda_wa = nn.Parameter(torch.zeros((1, n_feat, 1, 1)), requires_grad=True)

        self.norm_wa_mlp = LayerNorm2d(n_feat)
        self.mlp_wa = MLP(n_feat, int(n_feat * mlp_ratio), n_feat)

    def forward(self, x):
        x_norm = self.norm_ra(x)
        x = x + self.lambda_ra * self.ra(x_norm)
        x = self.mlp_ra(self.norm_ra_mlp(x)) + x

        x_norm = self.norm_wa(x)
        x = x + self.lambda_wa * self.wa(x_norm)
        x = self.mlp_wa(self.norm_wa_mlp(x)) + x

        return x


class PixBlock(nn.Module):
    def __init__(self, in_size, out_size=3, scale=2):
        super(PixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size * (2 ** scale), 1, 2)
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        return x


class Generator(nn.Module):
    def __init__(self,
                 img_channel=3,
                 base_channel=32,
                 middle_blk_num=12,
                 enc_blk_nums=[2, 2],
                 dec_blk_nums=[2, 2],
                 loss_fun=None):
        super(Generator, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=base_channel, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=base_channel, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.enc_kernel1 = KernelGuidedConv(output_dim=64)
        self.enc_kernel2 = KernelGuidedConv(output_dim=128)

        self.dec_kernel1 = KernelGuidedConv(output_dim=128)
        self.dec_kernel2 = KernelGuidedConv(output_dim=64)

        chan = base_channel
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        n_layer = 3
        self.middle_blks = nn.ModuleList([Block(n_layer=n_layer,
                                                n_feat=chan,
                                                num_heads=8,
                                                window_size=4,
                                                mlp_ratio=2)
                                          for i in range(int(middle_blk_num // n_layer))])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.loss_fun = loss_fun

    def forward(self, inp, k):
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            if x.shape[1] == 64:
                x = self.enc_kernel1(k, x)
            elif x.shape[1] == 128:
                x = self.enc_kernel2(k, x)
            encs.append(x)
            x = down(x)

        shortcut = x
        for blk in self.middle_blks:
            x = blk(x)
        x = x + shortcut

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            if x.shape[1] == 64:
                x = self.dec_kernel2(k, x)
            elif x.shape[1] == 128:
                x = self.dec_kernel1(k, x)

        x = self.ending(x)
        x = x + inp

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, norm=None):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm=None):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_filters))
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, norm),
            *discriminator_block(64, 128, norm),
            *discriminator_block(128, 256, norm),
            *discriminator_block(256, 512, norm),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class KernelGuidedConv(nn.Module):
    def __init__(self, kernel_shape=(512, 2, 2), output_dim=256):
        super(KernelGuidedConv, self).__init__()
        # Flatten模糊核输入
        self.input_dim = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]

        # 全连接层网络，分别生成权值向量 x^(k) 和偏置向量 b^(k)
        self.fc_weight = nn.Sequential(
            nn.Linear(self.input_dim, 1024),  # 第一层
            nn.ReLU(),
            nn.Linear(1024, 512),  # 第二层
            nn.ReLU(),
            nn.Linear(512, 256),  # 第二层
            nn.ReLU(),
            nn.Linear(256, output_dim)  # 最后一层，用于生成权值向量 x^(k)
        )

        # 卷积层
        self.conv = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, padding=1)

        # 可训练参数 α（逐通道），初始化为 1
        self.alpha = nn.Parameter(torch.ones(output_dim, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, blur_kernel, feature_map):
        # 展平模糊核
        blur_kernel = blur_kernel.view(blur_kernel.size(0), -1)  # [batch_size, 2048]

        # 全连接网络降维，生成权值向量 x^(k) 和偏置向量 b^(k)
        weight_vector = self.fc_weight(blur_kernel)  # 输出形状: [batch_size, 128/64]

        # 卷积操作
        conv_features = self.conv(feature_map)  # 输入特征图卷积，形状: [batch_size, 128/64, H, W]

        # 将权值和偏置扩展维度
        weight_vector = weight_vector.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 128/64, 1, 1]

        # 公式计算: r = r ⊙ (α + x^(k)) + b^(k)
        output = conv_features * (self.alpha + weight_vector) + weight_vector

        return self.relu(output)


if __name__ == '__main__':
    device = torch.device('cuda:1')
    ip = torch.ones((8, 3, 224, 224))
    k = torch.ones((8, 512, 2, 2))
    ip = ip.to(device)
    k = k.to(device)
    model = Generator(base_channel=64, middle_blk_num=12)
    model.to(device)
    start = time.time()
    op = model(ip, k)
    end = time.time()
    print(end - start)
    print(op.shape)
