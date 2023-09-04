import MinkowskiEngine as ME
import torch.nn as nn
import torch
import configs.settings as settings
from MinkowskiEngine.MinkowskiOps import (
    to_sparse,
    MinkowskiToSparseTensor,
    MinkowskiToDenseTensor
)
class MEConv(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3, full_minkowski=False):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension
        self.full_minkowski = full_minkowski
        self.features = self._make_layers(1, out_channel, embedding_channel)
        if(not full_minkowski):
            self.classifier = nn.Linear(4500, 10)
            # self.dense_shape = torch.Size([settings.modelconfig.getint("batch_size"), 1, 5, 30, 30])
        if(full_minkowski):
            self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
            self.classifier = ME.MinkowskiLinear(128, 10)
    def _make_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        channels = (128)
        layers += [ME.MinkowskiConvolution(in_channel, 128, kernel_size=3, dimension=self.D)]
        layers += [ME.MinkowskiInstanceNorm(128)]
        layers += [ME.MinkowskiReLU(inplace=True)]
        layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
        in_channel = 128
        out_channel = 64
        for d in range(3):
            layers += [ME.MinkowskiConvolution(in_channel, out_channel, kernel_size=3, dimension=self.D)]
            layers += [ME.MinkowskiInstanceNorm(out_channel)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel //= 2
            out_channel //= 2
        if(not self.full_minkowski):
            layers += [ME.MinkowskiConvolution(out_channel * 2, 1, kernel_size=1, dimension=self.D)]

        return nn.Sequential(*layers)

    def _make_equal_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        for d in range(3):
            layers += [ME.MinkowskiConvolution(in_channel, out_channel, kernel_size=3, dimension=self.D)]
            layers += [ME.MinkowskiInstanceNorm(out_channel)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel = 128
        if(not self.full_minkowski):
            layers += [ME.MinkowskiConvolution(out_channel * 2, 1, kernel_size=1, dimension=self.D)]

        return nn.Sequential(*layers)
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
                
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
    
    def forward(self, x: ME.TensorField):

        old_shape = x.shape
        out = x.sparse()
        out = self.features(out)
        if(not self.full_minkowski):
            min_coord, _ = out.C.min(0, keepdim=True)
            min_coord = min_coord[:, 1:].cpu()
            out = out.dense(shape=self.dense_shape, min_coordinate=min_coord)[0]
            out = out.view(settings.modelconfig.getint("batch_size"), -1)
            out = self.classifier(out)
        
        if(self.full_minkowski):
            out = self.global_avg_pool(out)
            out = self.classifier(out).F

        return out
    
class MEConvImage(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3, full_minkowski=False):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension
        self.full_minkowski = full_minkowski
        self.features = self._make_equal_layers(in_channel, out_channel, embedding_channel)
        self.weight_initialization()
        if(not full_minkowski):
            self.classifier = nn.Linear(100352, 10)
            # self.dense_shape = torch.Size([settings.modelconfig.getint("batch_size"), 128, 1, 28, 28])
        if(full_minkowski):
            self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
            self.classifier = ME.MinkowskiLinear(128, 10)
    def _make_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        channels = (128)
        layers += [ME.MinkowskiConvolution(in_channel, 128, kernel_size=3, dimension=self.D)]
        layers += [ME.MinkowskiInstanceNorm(128)]
        layers += [ME.MinkowskiReLU(inplace=True)]
        layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
        in_channel = 128
        out_channel = 64
        for d in range(3):
            layers += [ME.MinkowskiConvolution(in_channel, out_channel, kernel_size=3, dimension=self.D)]
            layers += [ME.MinkowskiInstanceNorm(out_channel)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel //= 2
            out_channel //= 2
        if(not self.full_minkowski):
            layers += [ME.MinkowskiConvolution(out_channel * 2, 1, kernel_size=1, dimension=self.D)]

        return nn.Sequential(*layers)

    def _make_equal_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        for d in range(3):
            layers += [ME.MinkowskiConvolution(in_channel, 128, kernel_size=3, dimension=self.D)]
            layers += [ME.MinkowskiInstanceNorm(128)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel = 128
        if(not self.full_minkowski):
            # pass
            layers += [ME.MinkowskiConvolution(in_channel, 128, kernel_size=1, dimension=self.D)]

        return nn.Sequential(*layers)
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
                
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
    
    def forward(self, x: ME.TensorField):

        old_shape = x.shape
        out = x.sparse()
        out = self.features(out)
        if(not self.full_minkowski):
            dense_shape = torch.Size([int(torch.max(x.C, dim=0).values[0].item())+1, 1, 1, 28, 28])
            min_coord, _ = out.C.min(0, keepdim=True)
            min_coord = min_coord[:, 1:].cpu()
            out = out.dense(shape=dense_shape, min_coordinate=min_coord)[0]
            out = out.view(-1, 100352)
            # out = out.F.reshape(-1, 2048)
            out = self.classifier(out)
        
        if(self.full_minkowski):
            out = self.global_avg_pool(out)
            out = self.classifier(out).F

        return out    

class MEPytorch(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension
        # self.mink = self._make_mink_layers(in_channel, 128)
        self.net_depth = 3
        self.pyt = self._make_pytorch_layers(in_channel, 128)
        layers_out = 10
        if(self.net_depth==3):
            layers_out = 1152
        elif(self.net_depth==4):
            layers_out = 128
        elif(self.net_depth==2):
            layers_out = 6272
        self.classifier = nn.Linear(layers_out, 10)

    
    def _make_pytorch_layers(self, in_channel, out_channel):
        layers = []
        net_depth = self.net_depth
        out_c = 128
        for d in range(net_depth):
            if(d == net_depth -1):
                out_c = out_channel
            # layers += [nn.Conv3d(in_channel, out_c, kernel_size=(1,3,3), padding=1)]
            layers += [nn.Conv3d(in_channel, out_c, kernel_size=(1,3,3), padding=1)]
            # layers += [nn.GroupNorm(out_c, out_c, affine=True)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            in_channel = 128
        return nn.Sequential(*layers)
    
    def forward(self, x: ME.TensorField):
        out = x.sparse()
        dense_shape = torch.Size([int(torch.max(x.C, dim=0).values[0].item())+1, 3, 1, 28, 28])
        min_coord, _ = out.C.min(0, keepdim=True)
        min_coord = min_coord[:, 1:].cpu()
        out = out.dense(shape=dense_shape, min_coordinate=min_coord)[0]
        out = self.pyt(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
class MEConvExp(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3, full_minkowski=False):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension
        self.full_minkowski = full_minkowski
        self.features = self._make_equal_layers(in_channel, 128, embedding_channel)
        self.weight_initialization()
        self.classifier = nn.Linear(100352, 10)

    def _make_equal_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        net_depth = 2
        out_c = 128
        for d in range(net_depth):
            if(d == net_depth-1):
                out_c = out_channel
            layers += [ME.MinkowskiConvolution(in_channel, out_c, kernel_size=3, dimension=self.D)]
            # layers += [ME.MinkowskiInstanceNorm(128)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel = 128

        return nn.Sequential(*layers)
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
                
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
    
    def forward(self, x: ME.TensorField):

        old_shape = x.shape
        out = x.sparse()
        out = self.features(out)
        if(not self.full_minkowski):
            dense_shape = torch.Size([int(torch.max(x.C, dim=0).values[0].item())+1, 128, 1, 28, 28])
            min_coord, _ = out.C.min(0, keepdim=True)
            max_coord, _ = out.C.max(0, keepdim=True)
            min_coord = min_coord[:, 1:].cpu()
            out = out.dense(shape=dense_shape, min_coordinate=min_coord)[0]
            out = out.view(-1, 100352)
            out = self.classifier(out)

        return out    

def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
    if is_minknet:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
        return ME.TensorField(
            coordinates=batch["coordinates"],
            features=batch["features"],
            device=device,
        )
    else:
        return batch["coordinates"].permute(0, 2, 1).to(device)