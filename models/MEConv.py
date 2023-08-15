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
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3, full_minkowski=True):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension ## 2 for img 3 for 3D
        # self.mlp = nn.Sequential(
        #         ME.MinkowskiLinear(3, 128, bias=False),
        #         ME.MinkowskiReLU(),
        #     )
        # self.global_sum_pool = ME.MinkowskiGlobalSumPooling()
        # self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        # self.global_sum = ME.MinkowskiGlobalSumPooling()
        self.full_minkowski = full_minkowski
        self.features = self._make_layers(3, out_channel, embedding_channel)
        if(not full_minkowski):
            self.classifier = nn.Linear(4500, 10)
            # self.sparse = MinkowskiToSparseTensor(False)
            self.dense_shape = torch.Size([settings.modelconfig.getint("batch_size"), 1, 5, 30, 30])
        # self.dense = MinkowskiToDenseTensor(shape=dense_shape)
        # self.dense = MinkowskiToDenseTensor()
        if(full_minkowski):
            self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
            self.classifier = ME.MinkowskiLinear(16, 10)
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
            ## Ignoring Batching for now
            layers += [ME.MinkowskiInstanceNorm(out_channel)]
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]
            in_channel //= 2
            out_channel //= 2
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
        # out = self.mlp(x)
        # coordinates = ME.dense_coordinates(x.shape)
        # out = x.sparse(coordinates=coordinates)
        old_shape = x.shape
        # print(x)
        # print("Before sparse:", x.shape)
        out = x.sparse()
        # print("After sparse:", out.shape)
        # print(out)
        # print("Sparse features shape", out.features.shape)
        # print(out.shape)
        out = self.features(out)
        # out = out.slice(x)
        # out = out.sparse()
        # print("After convs and pooling:",out.shape)
        # dense_temp = out.dense()[0]
        # print("Dense func",dense_temp.shape)
        # print("Dense after reshape", dense_temp.view(settings.modelconfig.getint("batch_size"), -1).shape)
        # print(out.features.shape)
        # out = self.global_avg_pool(out)
        # (out, a , b) = out.dense()
        # print(out.shape, a.shape, b.shape)
        # out = out.dense()
        # min_coordinate = out.C.min(0, keepdim=True)
        # max_coordinate = out.C.max(0, keepdim=True)
        # print(min_coordinate)
        # min_coordinate = min_coordinate[:, 1:]
        # print(settings.modelconfig.getint("batch_size"))
        # out = out.dense()[0]
        # out = self.dense(out)
        # print(out.shape)
        # # out_dense_reshape = out_dense.view(out_dense.size(0), -1)
        # out_avg = self.global_avg_pool(out)
        # out_max = self.global_max_pool(out)
        # print("Before global avg", out.shape, " After global average", out_avg.shape)
        # print("Before global max", out.shape, " After global max", out_max.shape)
        # print(out.shape)
        # print(out.C.min(0, keepdim=True).values)
        # print(out.C.max(0, keepdim=True))
        # print(out.F)
        if(not self.full_minkowski):
            min_coord, _ = out.C.min(0, keepdim=True)
            min_coord = min_coord[:, 1:].cpu()
            out = out.dense(shape=self.dense_shape, min_coordinate=min_coord)[0]
            out = out.view(settings.modelconfig.getint("batch_size"), -1)
            out = self.classifier(out)
        
        if(self.full_minkowski):
            out = self.global_avg_pool(out)
            out = self.classifier(out).F
        # print(out.shape)
        # print(out.shape)
        # out = self.global_sum(out)
        # out = self.global_sum_pool(out)

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