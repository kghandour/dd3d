import copy
import numpy as np
import torch
import time
import open3d as o3d
import os
from torch.utils.data import Dataset
from classification_model.shapepcd_set import (
    ShapeNetPCD,
    class_id,
    get_class_name_from_id,
    minkowski_collate_fn,
)
from classification_model.train_val_split import TRAIN_DICT, VAL_DICT


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit("loop hyper-parameters are not defined for %d ipc" % ipc)
    return outer_loop, inner_loop


def get_rand_cad(c, n, indices_cls, CAD_list):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_cls[c])[:n]
    return CAD_list[idx_shuffle]


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def get_cad_points(path, num_points):
    pcd = o3d.io.read_point_cloud(path)
    voxel_sz = 0.025
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_sz)
    xyz = np.asarray(downpcd.points)
    np.random.shuffle(xyz)
    xyz = xyz[:num_points]
    return xyz.to(torch.float32)


def save_cad(cad_list, config, directory, iteration):
    cad_list_copy = cad_list.clone().detach().cpu().numpy()
    for i, cad in enumerate(cad_list_copy):
        cad[cad < -1] = -1
        cad[cad > 1] = 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(cad))
        ipc = config.getint("n_cad_per_class", 1)
        name = get_class_name_from_id(i // ipc)
        o3d.io.write_point_cloud(
            directory
            + "/"
            + name
            + "_"
            + str(iteration)
            + ".ply",
            pcd,
        )


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


def list_average(lst):
    return sum(lst) / len(lst)


class SyntheticTensorDataset(Dataset):
    def __init__(self, cad, labels):
        self.cad = cad
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]
        xyz = self.cad[index]
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.cad.shape[0]


class RealTensorDataset(Dataset):
    def __init__(self, cad, labels):
        self.cad = cad
        self.labels = labels

    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.cad[index])
        xyz = np.asarray(pcd.points)
        xyz = xyz[:2048]
        label = self.labels[index]
        xyz = torch.from_numpy(xyz)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }

    def __len__(self):
        return self.cad.shape[0]


def get_real_cad_paths_and_labels(
    data_root, mode="train", cls_name=None, desired_amount=-1
):
    cad_path, labels = [], []

    # Check if dataset path exists
    assert os.path.exists(data_root), f"{data_root} does not exist"

    ## Are you looking for a certain class? Will require to repeat call for each class
    if cls_name is not None:
        if mode == "train":
            for model in TRAIN_DICT[cls_name]:
                labels.append(
                    class_id[cls_name]
                )  ## Will need to change label count afterwards
                cad_path.append(model)
        else:
            for model in VAL_DICT[cls_name]:
                labels.append(class_id[cls_name])
                cad_path.append(model)
        return cad_path, labels

    ## Do you want all class paths
    if mode == "train":
        for key in TRAIN_DICT.keys():
            for model in TRAIN_DICT[key]:
                labels.append(class_id[key])
                cad_path.append(model)

    else:
        for key in VAL_DICT.keys():
            for model in VAL_DICT[key]:
                labels.append(class_id[key])
                cad_path.append(model)

    return cad_path, labels


def create_val_loader_and_list(def_conf, classes_to_distill):
    cad_all_path, labels_all = [], []
    if len(classes_to_distill) == 0:
        cad_all_path, labels_all = get_real_cad_paths_and_labels(
            data_root=def_conf.get("shapenet_path"), mode="val"
        )
    else:
        for cls_name in classes_to_distill:
            cad_path_temp, labels_tmp = get_real_cad_paths_and_labels(
                data_root=def_conf.get("shapenet_path"), mode="val", cls_name=cls_name
            )
            cad_all_path.extend(cad_path_temp)
            labels_all.extend(labels_tmp)

    val_set = ShapeNetPCD(
        phase="val",
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        for_distillation=True,
        cls_list=classes_to_distill,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True
    )

    return np.array(cad_all_path), torch.from_numpy(np.array(labels_all)).type(torch.LongTensor), val_loader


## TODO: Add REAL Initialization.
def initalize_synthetic_tensors(
    num_classes,
    ipc,
    device,
    num_points=2048,
    initialization="random",
    classes_to_distill=[],
):
    ## Initialize synthetic cad. Creates a cube [-1, 1] with random points scattered.

    if len(classes_to_distill) > 0:
        cad_syn = torch.rand(
            size=(len(classes_to_distill) * ipc, num_points, 3), device=device
        )
        cad_syn = ((-1-1)*cad_syn + 1).requires_grad_()
        label_syn = torch.tensor(np.array([np.ones(ipc, dtype=int)*class_id[i] for i in classes_to_distill]), dtype=torch.long, requires_grad=False, device=device).view(-1)

    else:
        cad_syn = torch.rand(size=(num_classes*ipc, num_points, 3), device=device)
        cad_syn = ((-1-1)*cad_syn + 1).requires_grad_()
        label_syn = torch.tensor(np.array([np.ones(ipc, dtype=int)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1)

    return cad_syn, label_syn


def create_loader_for_synthetic_cad(cad_syn_tensor, label_syn_tensor, make_copy=True, batch_size=4):
    if(make_copy):
        cad_syn_eval_tensor, label_syn_eval_tensor = cad_syn_tensor.clone().detach(), label_syn_tensor.clone().detach()

    else:
        cad_syn_eval_tensor = cad_syn_tensor
        label_syn_eval_tensor = label_syn_tensor

    syn_ds = SyntheticTensorDataset(cad_syn_eval_tensor, label_syn_eval_tensor)
    syn_loader = torch.utils.data.DataLoader(
        syn_ds,
        batch_size=batch_size,
        collate_fn=minkowski_collate_fn,
        drop_last=False,
    )
    return syn_loader


def populate_classification_metrics_dict(classification_metrics_dict, single_eval_dict):
    classification_metrics_dict["log_acc_train"].append(single_eval_dict["acc_train"])
    classification_metrics_dict["log_acc_test"].append(single_eval_dict["acc_val"])
    classification_metrics_dict["log_time_train"].append(single_eval_dict["time_train"])
    classification_metrics_dict["log_loss_train"].append(single_eval_dict["loss_train"])
    classification_metrics_dict["log_loss_val"].append(single_eval_dict["loss_val"])
    return classification_metrics_dict

def log_classification_metrics_and_reset(logging, summary_writer, classification_metrics_dict, iteration):
    if(logging):
        summary_writer.add_scalar(
            "Classifier/Train_Accuracy", list_average(classification_metrics_dict["log_acc_train"]), iteration
        )
        summary_writer.add_scalar(
            "Classifier/Train_Time", list_average(classification_metrics_dict["log_time_train"]), iteration
        )
        summary_writer.add_scalar(
            "Classifier/Train_Loss", list_average(classification_metrics_dict["log_loss_train"]), iteration
        )
        summary_writer.add_scalar(
            "Classifier/Val_Accuracy", list_average(classification_metrics_dict["log_acc_test"]), iteration
        )
        summary_writer.add_scalar(
            "Classifier/Val_Loss", list_average(classification_metrics_dict["log_loss_val"]), iteration
        )
    classification_metrics_dict = {}
    classification_metrics_dict["log_acc_train"]=[]
    classification_metrics_dict["log_acc_test"]=[]
    classification_metrics_dict["log_time_train"]=[]
    classification_metrics_dict["log_loss_train"]=[]
    classification_metrics_dict["log_loss_val"]=[]
