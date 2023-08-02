import MinkowskiEngine as ME
import torch.nn as nn

class MEConv(ME.MinkowskiNetwork):
    def __init__(self, in_channel, out_channel, embedding_channel=1024, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.D = dimension ## 2 for img 3 for 3D
        self.mlp = nn.Sequential(
                ME.MinkowskiLinear(3, 128, bias=False),
                ME.MinkowskiLeakyReLU(),
            )
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.features = self._make_layers(128, out_channel, embedding_channel)
        self.classifier = ME.MinkowskiLinear(128, 10)
    def _make_layers(self, in_channel, out_channel, embedding_channel):
        layers = []
        channels = (128)
        for d in range(3):
            layers += [ME.MinkowskiConvolution(in_channel, 128, kernel_size=3, dimension=self.D)]
            ## Ignoring Batching for now
            layers += [ME.MinkowskiReLU(inplace=True)]
            layers += [ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension= self.D)]

        return nn.Sequential(*layers)
    
    def forward(self, x: ME.TensorField):
        out = self.mlp(x)
        out = out.sparse()
        out = self.features(out)
        out = self.global_avg_pool(out)
        out = self.classifier(out)
        return out.F
    
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