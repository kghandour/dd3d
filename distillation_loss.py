import configs.settings as settings
from tqdm import tqdm
import models.pointnet2_ssg_wo_normals.provider as provider
import torch
import numpy as np


def get_distillation_loss(distillation_network, distillation_criterion, dataloader, ):
    losses = []
    for batch_id, (points, target) in enumerate(dataloader, 0):
        # points = points.data.numpy()
        # points = provider.random_point_dropout(points)
        # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        # points = torch.Tensor(points)
        points = points.transpose(2, 1)

        points, target = points.to(settings.device), target.to(settings.device)

        pred, trans_feat = distillation_network(points)
        loss = distillation_criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]
        losses.append(loss)
    return sum(losses)/len(losses)

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = "do nothing"
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(
        1
        - torch.sum(gwr * gws, dim=-1)
        / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    )
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, dis_metric, device):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == "ours":
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("unknown distance function: %s" % dis_metric)

    return dis