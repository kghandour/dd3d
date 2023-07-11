import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class MinkowskiDistill(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        channels=(32, 48, 64, 96, 128),
        embedding_channel=1024,
        D=3,
        num_points=2048,
        net_depth=3,
        activation="relu",
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.activation = activation
        self.D = D
        self.network_initialization(
            in_channel,
            out_channel,
            channels,
            embedding_channel,
            kernel_size=3,
            D=D
        )
        self.weight_initialization()

    def getActivation(self):
        if self.activation == "relu":
            return ME.MinkowskiReLU()
        elif self.activation == "leakyrelu":
            return ME.MinkowskiLeakyReLU()
        else:
            return ME.MinkowskiTanh()

    def mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            self.getActivation(),
        )

    def conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                dimension=self.D,
            ),
            self.getActivation(),
        )

    def network_initialization(self, in_channel, out_channel, channels, embedding_channel, kernel_size, D=3):
        self.init_FC = self.mlp_block(in_channel, channels[0])
        shape_feat = [in_channel]
        self.conv1 = self.conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1
        )
        self.conv2 = self.conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=1
        )

        self.conv3 = self.conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=1,
        )

        self.conv4 = self.conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=1,
        )

        self.conv5 = nn.Sequential(
            self.conv_block(
                channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=1,
            ),
            self.conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=1,
            ),
            self.conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=1,
            ),
        )

        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()

        self.pool = ME.MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.final = nn.Sequential(
            self.mlp_block(embedding_channel, 512),
            ME.MinkowskiLinear(512, out_channel, bias=False),
        ) 

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)



    def forward(self, x: ME.TensorField):
        x = self.init_FC(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        # min_coordinate, _ = y.C.min(0, keepdim=True)
        # min_coordinate = min_coordinate[:, 1:].cpu()
        # # min_coordinate = min_coordinate.type(torch.int32)
        # y_dense = y.dense(min_coordinate=min_coordinate)[0]
        # y_dense_shape = y_dense.view(y_dense.size(0),-1)
        # print(y_dense_shape.shape)
        y4 = self.pool(y)
        # print(y4.shape)
        # x1 = y1.slice(x)
        # print(x1.shape)
        # x2 = y2.slice(x)
        # x3 = y3.slice(x)
        # x4 = y4.slice(x)

        # x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(y4)
        # x1 = self.global_avg_pool(y)
        # x2 = self.global_avg_pool(y)
        y_out = self.global_max_pool(y)
        # y_out_2 = self.global_avg_pool(y)
        # return self.tfinal(y_dense_shape)
        return self.final(y_out).F