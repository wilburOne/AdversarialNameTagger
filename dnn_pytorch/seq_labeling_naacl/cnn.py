import torch
import torch.nn as nn


class MultiLeNetConv2dLayer(nn.Module):
    def __init__(self, kernel_shape, pool_sizes):
        super(MultiLeNetConv2dLayer, self).__init__()

        num_conv = len(kernel_shape)
        in_channels = [item[0] for item in kernel_shape]
        out_channels = [item[1] for item in kernel_shape]
        kernel_size = [item[2] for item in kernel_shape]

        self.conv_nets = []
        self.max_pool2d = []
        for i in range(num_conv):
            conv = nn.Conv2d(in_channels[i], int(out_channels[i]), kernel_size[i])
            self.conv_nets.append(conv)

            max_pool2d = nn.MaxPool2d(pool_sizes[i])
            self.max_pool2d.append(max_pool2d)
        self.conv_nets = nn.ModuleList(self.conv_nets)
        self.max_pool2d = nn.ModuleList(self.max_pool2d)

    def forward(self, input):
        conv_out = []
        input = torch.unsqueeze(input, 1)
        for conv in self.conv_nets:
            conv_out.append(conv(input))

        pooling_out = []
        for i, pool in enumerate(self.max_pool2d):
            # squeeze the last two dimensions
            after_pool = pool(conv_out[i]).squeeze(2).squeeze(2)

            pooling_out.append(after_pool)

        return pooling_out