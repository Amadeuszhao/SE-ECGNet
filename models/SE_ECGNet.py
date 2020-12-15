import torch
import torch.nn as nn


class SE_Module(nn.Module):

    def __init__(self, in_channels, ratio=16, dim=2):
        super(SE_Module, self).__init__()
        self.dim = dim
        if self.dim == 1:
            self.squeeze = nn.AdaptiveAvgPool1d(1)
        else:
            self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // ratio, out_features=in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x => [b, c, h, w] / x => [b, c, l]
        identity = x

        out = self.squeeze(x)
        out = out.reshape(out.shape[0], out.shape[1])
        scale = self.excitation(out)
        if self.dim == 1:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1)
        else:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1, 1)

        return identity * scale.expand_as(identity)



class ResBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, downsample=None, num_conv=None):
        super(ResBlock1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.SE1 = SE_Module(in_channels=out_channels, dim=1)
        self.bn4 = nn.BatchNorm1d(num_features=out_channels)
        self.conv4 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn5 = nn.BatchNorm1d(num_features=out_channels)
        self.conv5 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn6 = nn.BatchNorm1d(num_features=out_channels)
        self.conv6 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.SE2 = SE_Module(in_channels=out_channels, dim=1)
        self.downsample = downsample
        self.dropout =nn.Dropout(.2)




    def forward(self, x):
        identity = x  # x => [b, 256, 310]

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.SE1(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        identity = out

        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv4(out)

        out = self.bn5(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv5(out)

        out = self.bn6(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv6(out)

        out = self.SE2(out)


        out += identity
        out = self.relu(out)

        return out


class ResBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, downsample=None, num_conv=1):
        super(ResBlock2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.SE = SE_Module(in_channels=out_channels)
        self.downsample = downsample
        self.num_conv = num_conv
        self.dropout = nn.Dropout(.2)
        if self.num_conv == 3:
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=out_channels)
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False)


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        if self.num_conv == 3:
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv3(out)
            # print(out.shape)

        out = self.SE(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_ECGNet(nn.Module):

    def __init__(self, struct=[(1, 3), (1, 5), (1, 7)], num_classes=34):
        super(ECGNet, self).__init__()
        self.struct = struct
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 50), stride=(1, 2), padding=(0, 0),
                              bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=256 * len(struct), out_features=num_classes)
        self.block1 = self._make_layer(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 2),
                                       block=ResBlock2d, blocks=3, padding=(0, 7))

        self.block2_list = nn.ModuleList()
        self.block3_list = nn.ModuleList()

        for i, kernel_size in enumerate(self.struct):
            block2 = self._make_layer(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1, 1),
                                      block=ResBlock2d, blocks=4, padding=(0, 1 * (i + 1)))
            block3 = self._make_layer(in_channels=256, out_channels=256, kernel_size=kernel_size[1], stride=2,
                                      block=ResBlock1d, blocks=4, padding=1 * (i + 1))
            self.block2_list.append(block2)
            self.block3_list.append(block3)

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, block, blocks, padding=(0, 0)):
        layers = []
        num_conv = 1
        if blocks == 4:
            num_conv = 3
        downsample = None
        if blocks == 3:
            downsample = nn.Sequential(
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 2),
                          padding=(0, 0))
            )
        if block == ResBlock1d:
            downsample = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=2,
                          padding=0)
            )
        for _ in range(blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, downsample=downsample, num_conv=num_conv))

        return nn.Sequential(*layers)



    def forward(self, x, info=None):
        out = x.unsqueeze(1)
        out = self.conv(out)  # x => [b, 32, 8, 2476]
        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)  # x => [b, 32, 8, 310]

        out_sep = []
        for i in range(len(self.struct)):
            sep = self.block2_list[i](out)  # x => [b, 32, 8, 310]
            sep = sep.reshape(sep.shape[0], -1, sep.shape[3])  # x => [b, 256, 310]
            sep = self.block3_list[i](sep)  # x => [b, 256, 20]
            sep = self.avgpool(sep)  # x => [b, 256, 1]
            sep = sep.reshape(sep.shape[0], sep.shape[1])
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=1)
        if info != None:
            out = torch.cat([out, info], dim=1)
        out = self.fc(out)

        return out





if __name__ == '__main__':
    input = torch.randn(16, 8, 5000)
    info = torch.randn(16, 2)
    SE_ECGNet = SE_ECGNet()
    output = SE_ECGNet(input)
    print(output.shape)
    # print(output.shape)