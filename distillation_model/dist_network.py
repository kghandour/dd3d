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
        embedding_channel=1024,
        channels=(256, 512, 1024, 2048),
        D=3,
        num_points = 2048,
        net_depth = 3
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.FC = self.FCLayer(in_channel=in_channel, out_channel=channels[0])
        self.features, shape_feat = self.network_initialization(
            channels[0],
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
            num_points=num_points,
            net_depth=net_depth
        )
        num_feat = shape_feat[0]*shape_feat[1]
        self.classifier = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel * 2, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiLeakyReLU(),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )
        self.weight_initialization()

    def FCLayer(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
        num_points = 2048,
        net_depth=3
    ):
        shape_feat = [in_channel, num_points]
        layers = []
        ## TODO: Need to linear layer then sparse it.
        for d in range(net_depth):
            layers += nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                channels[d+1],
                kernel_size=kernel_size,
                stride=1,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(channels[d+1]),
            ME.MinkowskiLeakyReLU(),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
            )
            in_channel = channels[d+1]
            shape_feat[0] = channels[d+1]
            shape_feat[1] //= 2
        
        return nn.Sequential(*layers), shape_feat
            
            
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        out = self.FC(x)
        out = out.sparse()
        out = self.features(out)
        out = self.classifier(out).F
        return out