import torch
import torch.nn as nn
import torch.nn.modules.utils as torch_utils
from collections import namedtuple

ConvLayerConfig = namedtuple('LayerConfig', 'in_channels out_channels kernel_size padding pool')


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, pool=True, relu=True, bn=False,
                 dropout_rate=0.0):
        super(Conv, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=not bn)]  # 如果使用BN，则禁用bias
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(Residual, self).__init__()
        # 预激活结构
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(in_channels, out_channels, 1,
                              bias=False) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        # 预激活结构
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.dropout(out)

        return out + identity


class Hourglass(nn.Module):
    def __init__(self, depth, nc, expansion, dropout_rate=0.1):
        super(Hourglass, self).__init__()
        self.depth = depth
        nc_expanded = nc + expansion
        self.up1 = Residual(nc, nc, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(nc, nc_expanded, dropout_rate=dropout_rate)
        if self.depth > 1:
            self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion, dropout_rate=dropout_rate)
        else:
            self.low2 = Residual(nc_expanded, nc_expanded, dropout_rate=dropout_rate)
        self.low3 = Residual(nc_expanded, nc, dropout_rate=dropout_rate)
        # 改为双线性上采样
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 添加特征融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        up1 = self.up1(x)
        pool = self.pool(x)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        low3 = self.dropout(low3)
        up2 = self.up2(low3)

        # 使用特征融合层而不是简单的加法
        combined = torch.cat([up1, up2], dim=1)
        return self.fuse(combined)


class HourglassBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(HourglassBackbone, self).__init__()
        self.layers = nn.Sequential(
            Conv(in_channels, 64, kernel_size=7, stride=2, pool=False,
                 padding=3, relu=True, bn=True, dropout_rate=dropout_rate),
            Residual(64, 128, dropout_rate=dropout_rate),
            nn.MaxPool2d(2, 2),
            Residual(128, 128, dropout_rate=dropout_rate),
            Residual(128, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


class RefineBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(RefineBackbone, self).__init__()
        self.layers = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0,
                 pool=True, dropout_rate=dropout_rate),
            Conv(in_channels=6, out_channels=out_channels, kernel_size=5, padding=0,
                 pool=True, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


class RefineBackboneKP(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(RefineBackboneKP, self).__init__()
        self.layers = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0,
                 pool=True, dropout_rate=dropout_rate),
            Conv(in_channels=6, out_channels=16, kernel_size=5, padding=0,
                 pool=True, dropout_rate=dropout_rate),
            Conv(in_channels=16, out_channels=out_channels, kernel_size=5, padding=0,
                 pool=True, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.layers(x)


def conv_module(layer_configs, dropout_rate=0.1):
    layers = []
    for layer_config in layer_configs:
        layers.append(Conv(in_channels=layer_config.in_channels,
                           out_channels=layer_config.out_channels,
                           kernel_size=layer_config.kernel_size,
                           padding=layer_config.padding,
                           pool=layer_config.pool,
                           dropout_rate=dropout_rate))
    return nn.Sequential(*layers)


def staged_conv_module(staged_layer_configs, dropout_rate=0.1):
    stage2layers = []
    for layer_configs, count in staged_layer_configs:
        for _ in range(count):
            stage2layers.append(conv_module(layer_configs, dropout_rate=dropout_rate))
    return nn.ModuleList(stage2layers)


def fc_module(init_in_features, final_out_features, inner_layer_dims, relu=True, dropout_rate=0.1):
    layers = []
    for i in range(len(inner_layer_dims)):
        in_features = init_in_features if i == 0 else inner_layer_dims[i - 1]
        out_features = final_out_features if i == (len(inner_layer_dims) - 1) else inner_layer_dims[i]
        layers.append(nn.Linear(in_features=in_features, out_features=out_features))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0 and i < len(inner_layer_dims) - 1:  # 不在最后一层添加dropout
            layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


# noinspection PyProtectedMember
def get_conv2d_layer_output_shape(in_dim, kernel_size, stride, padding, dilation=1):
    in_dim = torch_utils._pair(in_dim)
    kernel_size = torch_utils._pair(kernel_size)
    stride = torch_utils._pair(stride)
    padding = torch_utils._pair(padding)
    dilation = torch_utils._pair(dilation)
    out_dim_0 = (in_dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_dim_1 = (in_dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return out_dim_0, out_dim_1


def get_conv_module_output_shape(input_dim, module):
    dim = input_dim
    for module_layer in module:
        if isinstance(module_layer, Conv):
            for layer in module_layer.layers:
                if isinstance(layer, nn.Conv2d):
                    dim = get_conv2d_layer_output_shape(dim, layer.kernel_size, layer.stride, layer.padding)
                elif isinstance(layer, nn.MaxPool2d):
                    dim = (int(dim[0] / 2),
                           int(dim[1] / 2))
    return dim
# import torch.nn as nn
# import torch.nn.modules.utils as torch_utils
# from collections import namedtuple
# import math
#
# ConvLayerConfig = namedtuple('LayerConfig', 'in_channels out_channels kernel_size padding pool')
#
#
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, channels, num_heads=8):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = channels // num_heads
#
#         assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
#
#         self.query = nn.Conv2d(channels, channels, kernel_size=1)
#         self.key = nn.Conv2d(channels, channels, kernel_size=1)
#         self.value = nn.Conv2d(channels, channels, kernel_size=1)
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.scale = 1.0 / math.sqrt(self.head_dim)
#
#         self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#
#         # Generate queries, keys, values
#         Q = self.query(x).view(batch_size, self.num_heads, self.head_dim, height * width)
#         K = self.key(x).view(batch_size, self.num_heads, self.head_dim, height * width)
#         V = self.value(x).view(batch_size, self.num_heads, self.head_dim, height * width)
#
#         # Transpose for matrix multiplication
#         Q = Q.transpose(2, 3)  # (batch_size, num_heads, height*width, head_dim)
#         K = K.transpose(2, 3).transpose(2, 3)  # (batch_size, num_heads, head_dim, height*width)
#         V = V.transpose(2, 3)  # (batch_size, num_heads, height*width, head_dim)
#
#         # Calculate attention scores
#         attention_scores = torch.matmul(Q, K) * self.scale
#         attention_probs = self.softmax(attention_scores)
#
#         # Apply attention to values
#         out = torch.matmul(attention_probs, V)
#         out = out.transpose(2, 3).contiguous().view(batch_size, channels, height, width)
#
#         # Final linear layer
#         out = self.out_conv(out)
#         return out
#
#
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, pool=True, relu=True, bn=False):
#         super(Conv, self).__init__()
#         layers = [nn.Conv2d(in_channels=in_channels,
#                             out_channels=out_channels,
#                             kernel_size=kernel_size,
#                             stride=stride,
#                             padding=padding)]
#         if pool:
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
#         if bn:
#             layers.append(nn.BatchNorm2d(out_channels))
#         if relu:
#             layers.append(nn.ReLU(inplace=True))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, use_attention=False, num_heads=8):
#         super(Residual, self).__init__()
#         self.use_attention = use_attention
#
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(out_channels // 2)
#         self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels // 2)
#         self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#
#         # Add attention mechanism if requested
#         if use_attention:
#             self.attention = MultiHeadSelfAttention(out_channels, num_heads)
#
#         self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         residual = x if self.skip is None else self.skip(x)
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#
#         # Apply attention if enabled
#         if self.use_attention:
#             out = self.attention(out)
#
#         out += residual
#         return out
#
#
# class Hourglass(nn.Module):
#     def __init__(self, depth, nc, expansion, use_attention=False, num_heads=8):
#         super(Hourglass, self).__init__()
#         self.depth = depth
#         self.use_attention = use_attention
#         nc_expanded = nc + expansion
#
#         self.up1 = Residual(nc, nc, use_attention=use_attention, num_heads=num_heads)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.low1 = Residual(nc, nc_expanded, use_attention=use_attention, num_heads=num_heads)
#
#         if self.depth > 1:
#             self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion,
#                                   use_attention=use_attention, num_heads=num_heads)
#         else:
#             self.low2 = Residual(nc_expanded, nc_expanded,
#                                  use_attention=use_attention, num_heads=num_heads)
#
#         self.low3 = Residual(nc_expanded, nc, use_attention=use_attention, num_heads=num_heads)
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
#
#     def forward(self, x):
#         up1 = self.up1(x)
#         pool = self.pool(x)
#         low1 = self.low1(pool)
#         low2 = self.low2(low1)
#         low3 = self.low3(low2)
#         up2 = self.up2(low3)
#         return up1 + up2
#
#
# class HourglassBackbone(nn.Module):
#     def __init__(self, in_channels, out_channels, use_attention=False, num_heads=8):
#         super(HourglassBackbone, self).__init__()
#         self.layers = nn.Sequential(
#             Conv(in_channels, 64, kernel_size=7, stride=2, pool=False,
#                  padding=3, relu=True, bn=True),
#             Residual(64, 128, use_attention=use_attention, num_heads=num_heads),
#             nn.MaxPool2d(2, 2),
#             Residual(128, 128, use_attention=use_attention, num_heads=num_heads),
#             Residual(128, out_channels, use_attention=use_attention, num_heads=num_heads)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# class StackedHourglass(nn.Module):
#     def __init__(self, in_channels, out_channels, num_stacks=4, use_attention=True, num_heads=8):
#         super(StackedHourglass, self).__init__()
#         self.num_stacks = num_stacks
#         self.use_attention = use_attention
#
#         # Initial feature extraction
#         self.init_conv = nn.Sequential(
#             Conv(in_channels, 64, kernel_size=7, stride=2, padding=3, pool=False, bn=True),
#             Residual(64, 128),
#             nn.MaxPool2d(2, 2),
#             Residual(128, 128),
#             Residual(128, 256)
#         )
#
#         # Create multiple hourglass modules
#         self.hourglasses = nn.ModuleList()
#         for _ in range(num_stacks):
#             self.hourglasses.append(
#                 Hourglass(4, 256, 128, use_attention=use_attention, num_heads=num_heads)
#             )
#
#         # Feature processing between hourglasses
#         self.inter_features = nn.ModuleList()
#         for _ in range(num_stacks - 1):
#             self.inter_features.append(
#                 nn.Sequential(
#                     Residual(256, 256, use_attention=use_attention, num_heads=num_heads),
#                     Conv(256, 256, kernel_size=1, stride=1, padding=0, pool=False)
#                 )
#             )
#
#         # Output layers for each stack
#         self.output_layers = nn.ModuleList()
#         for _ in range(num_stacks):
#             self.output_layers.append(
#                 nn.Sequential(
#                     Residual(256, 256, use_attention=use_attention, num_heads=num_heads),
#                     Conv(256, out_channels, kernel_size=1, stride=1, padding=0, pool=False, relu=False, bn=False)
#                 )
#             )
#
#         # Heatmap aggregation
#         self.heatmap_aggregation = nn.ModuleList()
#         for _ in range(num_stacks - 1):
#             self.heatmap_aggregation.append(
#                 Conv(out_channels, 256, kernel_size=1, stride=1, padding=0, pool=False)
#             )
#
#     def forward(self, x):
#         # Initial feature extraction
#         features = self.init_conv(x)
#
#         outputs = []
#         for i in range(self.num_stacks):
#             # Process through hourglass
#             hourglass_out = self.hourglasses[i](features)
#
#             # Generate output
#             output = self.output_layers[i](hourglass_out)
#             outputs.append(output)
#
#             # Prepare for next stack if not the last one
#             if i < self.num_stacks - 1:
#                 # Process features for next stack
#                 features = self.inter_features[i](hourglass_out)
#
#                 # Add heatmap information to features
#                 heatmap_features = self.heatmap_aggregation[i](output)
#                 features = features + heatmap_features
#
#         return outputs