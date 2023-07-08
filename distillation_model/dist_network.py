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
        embedding_channel=128,
        channels=(1024, 512,256, 128),
        D=3,
        num_points = 2048,
        net_depth = 3
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.FC = self.FCLayer(in_channel=in_channel, out_channel=channels[0])
        self.features = self.network_initialization(
            channels[0],
            channels=channels,
            kernel_size=2,
            D=D,
            num_points=num_points,
            net_depth=net_depth
        )
        self.classifier = nn.Sequential(
            ME.MinkowskiLinear(128, out_channel, bias=False),
        )
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.weight_initialization()

    def FCLayer(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        channels,
        kernel_size,
        D=3,
        num_points = 2048,
        net_depth=3
    ):
        # shape_feat = [in_channel, num_points]
        layers = []
        for d in range(net_depth):
            layers += nn.Sequential(
                ME.MinkowskiConvolution(
                    in_channel,
                    channels[d+1],
                    kernel_size=kernel_size,
                    stride=1,
                    dimension=self.D,
                ),
                ME.MinkowskiLeakyReLU(),
                ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)
            )
            in_channel = channels[d+1]
        
        return nn.Sequential(*layers)
            
            
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

            if isinstance(m, ME.MinkowskiInstanceNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: ME.TensorField):
        out = self.FC(x)
        out = out.sparse()
        out = self.features(out)
        x1 = self.global_avg_pool(out)
        out = self.classifier(x1).F
        return out