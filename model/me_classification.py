# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.tensorboard import SummaryWriter
import torch
from me_network import criterion

def make_data_loader(phase, is_minknet, config):
    assert phase in ["train", "val", "test"]
    is_train = phase == "train"
    dataset = ModelNet40H5(
        phase=phase,
        transform=CoordinateTransformation(trans=config.translation)
        if is_train
        else CoordinateTranslation(config.test_translation),
        data_root="modelnet40_ply_hdf5_2048",
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=is_train,
        collate_fn=minkowski_collate_fn if is_minknet else stack_collate_fn,
        batch_size=config.batch_size,
    )

def train(net, device, config):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
    )
    print(optimizer)
    print(scheduler)

    train_iter = iter(make_data_loader("train", is_minknet, config))
    best_metric = 0
    net.train()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data_dict = train_iter.next()
        except StopIteration:
            train_iter = iter(make_data_loader("train", is_minknet, config))
            data_dict = train_iter.next()
        input = create_input_batch(
            data_dict, is_minknet, device=device, quantization_size=config.voxel_size
        )
        logit = net(input)
        loss = criterion(logit, data_dict["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            print(f"Iter: {i}, Loss: {loss.item():.3e}")

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )
            accuracy = test(net, device, config, phase="val")
            if best_metric < accuracy:
                best_metric = accuracy
            print(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()