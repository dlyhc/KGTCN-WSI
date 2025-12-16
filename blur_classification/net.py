import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelModule(nn.Module):  # 继承nn模块的Module类
    def __init__(self, inputs, ratio=16):  # self必写，inputs接收输入特征张量，ratio是通道衰减因子
        super(ChannelModule, self).__init__()  # 调用父类构造
        _, c, _, _ = inputs.size()  # 获取通道数
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # nn模块的自适应二维最大池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # nn模块的自适应二维平均池化
        self.share_liner = nn.Sequential(
            nn.Linear(c, c // ratio),
            nn.ReLU(),
            nn.Linear(c // ratio, c)
        )  # 这个共享全连接的3层和SEnet的一模一样，这里借助Sequential这个容器把这3个层整合在一起，方便forward函数去执行，直接调用share_liner(x)相当于直接执行了里面这3层
        self.sigmoid = nn.Sigmoid()  # nn模块的Sigmoid函数

    def forward(self, inputs):
        x = self.maxpool(inputs).view(inputs.size(0),
                                      -1)  # 对于输入特征张量，做完最大池化后再重塑形状，view的第一个参数inputs.size(0)表示第一维度，显然就是n；-1表示会自适应的调整剩余的维度，在这里就将原来的(n,c,1,1)调整为了(n,c*1*1)，后面才能送入全连接层（fc层）
        maxout = self.share_liner(x).unsqueeze(2).unsqueeze(
            3)  # 做完全连接后，再用unsqueeze解压缩，也就是还原指定维度，这里用了两次，分别还原2维度的h，和3维度的w
        y = self.avgpool(inputs).view(inputs.size(0), -1)
        avgout = self.share_liner(y).unsqueeze(2).unsqueeze(3)  # y走的平均池化路线的代码和x是一样的解释
        return self.sigmoid(maxout + avgout)  # 最后相加两个结果并作归一化


class SpatialModule(nn.Module):
    def __init__(self):
        super(SpatialModule, self).__init__()
        self.maxpool = torch.max
        self.avgpool = torch.mean
        # 和通道机制不一样！这里要进行的是在C这一个维度上求最大和平均，分别用的是torch库里的max方法和mean方法
        self.concat = torch.cat  # torch的cat方法，用于拼接两个张量
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1,
                              padding=3)  # nn模块的二维卷积，其中的参数分别是：输入通道（2），输出通道（1），卷积核大小（7*7），步长（1），灰度填充（3）
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        maxout, _ = self.maxpool(inputs, dim=1,
                                 keepdim=True)  # maxout接收特征点的最大值很好理解，为什么还要一个占位符？因为torch.max不仅返回张量最大值，还会返回索引，索引用不着所以直接忽略，dim=1表示在维度1（也就是nchw的c）上求最大值，keepdim=True表示要保持原来张量的形状
        avgout = self.avgpool(inputs, dim=1, keepdim=True)  # torch.mean则只返回张量的平均值，至于参数的解释和上面是一样的
        outs = self.concat([maxout, avgout],
                           dim=1)  # torch.cat方法，传入一个列表，将列表中的张量在指定维度，这里是维度1（也就是nchw的c）拼接，即n*1*h*w拼接n*1*h*w得到n*2*h*w
        outs = self.conv(outs)  # 卷积压缩上面的n*2*h*w，又得到n*1*h*w
        return self.sigmoid(outs)


class CBAM(nn.Module):
    def __init__(self, inputs):
        super(CBAM, self).__init__()
        self.channel_out = ChannelModule(inputs)  # 获得通道权重
        self.spatial_out = SpatialModule()  # 获得空间权重

    def forward(self, inputs):
        outs = self.channel_out(inputs) * inputs  # 先乘上通道权重
        return self.spatial_out(outs) * outs  # 在乘完通道权重的基础上再乘上空间权重

# 定义Channel Attention模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 定义网络结构
class DefocusedImageClassificationNet(nn.Module):
    def __init__(self, num_classes=5):
        super(DefocusedImageClassificationNet, self).__init__()

        # 特征提取块1
        self.block1 = self._make_block(3, 32, kernel_size=5, stride=1, pool_kernel=2)
        # 特征提取块2
        self.block2 = self._make_block(32, 64, kernel_size=5, stride=1, pool_kernel=2)
        # 特征提取块3
        self.block3 = self._make_block(64, 128, kernel_size=5, stride=1, pool_kernel=2)
        # 特征提取块4
        self.block4 = self._make_block(128, 128, kernel_size=5, stride=1, pool_kernel=2)

        # 1×1卷积进行特征融合和维度压缩
        self.conv1x1 = nn.Conv2d(128, 32, kernel_size=1)

        # 全连接层
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def _make_block(self, in_channels, out_channels, kernel_size, stride, pool_kernel):
        """构造单个特征提取块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            ChannelAttention(out_channels),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=2)
        )

    def forward(self, x):
        # 特征提取块
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 1×1卷积进行特征融合和维度压缩
        x = self.conv1x1(x)

        # 展平并通过全连接层
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 测试网络
if __name__ == "__main__":
    model = DefocusedImageClassificationNet(num_classes=5)

    # 输入一个随机张量
    input_tensor = torch.randn(1, 3, 224, 224)  # 假设输入图片尺寸为256x256
    output = model(input_tensor)
    print(output.shape)  # 输出类别数量
