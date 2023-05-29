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
import time

import torch.optim as optim
import MinkowskiEngine as ME
from torch.utils.tensorboard import SummaryWriter
import torch
from classification_model.me_network import criterion
import sklearn.metrics as metrics
import numpy as np

def create_input_batch(batch, device="cuda", quantization_size=0.05):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        device=device,
    )
   
def test(net, device, config, val_loader, phase="val"):

    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input = create_input_batch(
                batch,
                device=device,
                quantization_size=float(config.get("voxel_size")),
            )
            logit = net(input)
            pred = torch.round(torch.sigmoid(logit))
            labels.append(batch["labels"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            torch.cuda.empty_cache()
    return metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds))

def train(net, device, config, writer, train_dataloader, val_loader):
    optimizer = optim.SGD(
        net.parameters(),
        lr=float(config.get("lr")),
        momentum=0.9,
        weight_decay=float(config.get("weight_decay")),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config.get("max_steps")),
    )
    epoch_ct = 0
    print(optimizer)
    print(scheduler)
    train_iter = iter(train_dataloader)
    best_metric = 0
    net.train()
    for i in range(int(config.get("max_steps"))):
        optimizer.zero_grad()
        startTime = time.time()
        try:
            data_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            data_dict = next(train_iter)
        input = create_input_batch(
            data_dict, device=device, quantization_size=float(config.get("voxel_size"))
        )
        logit = net(input)
        loss = criterion(logit, data_dict["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        endTime = time.time()

        if i % int(config.get("stat_freq")) == 0 and i > 0:
            print(f"Iteration: {i}, Loss: {loss.item():.3e}")
            writer.add_scalar('loss/training_iter', loss.item(), i) 
            writer.add_scalar('time/training_iter', ((endTime - startTime)*1000), i)

        if i % len(train_dataloader) == 0 and i > 0:
            epoch_ct +=1
            writer.add_scalar('loss/training_epoch', loss.item(), epoch_ct) 
            startTime_val = time.time()
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.get("exp_name")+config.get("binary_class_name")+"_overfit_"+str(config.getboolean("overfit_1"))+".model",
            )
            accuracy = test(net, device, config, phase="val", val_loader=val_loader)
            endTime_val = time.time()
            if best_metric < accuracy:
                best_metric = accuracy
            writer.add_scalar('accuracy/val', accuracy, epoch_ct) 
            writer.add_scalar('time/validation', ((endTime_val - startTime_val)*1000), epoch_ct)
            print(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()