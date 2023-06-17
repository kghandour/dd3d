import copy
import numpy as np
import torch
import time
import open3d as o3d
from torch.utils.data import Dataset
from classification_model.shapepcd_set import class_id, get_class_name_from_id

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
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def get_rand_cad(c, n, indices_cls, CAD_list): # get random n images from class c
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

def save_cad(cad_list, config):
    cad_list_copy = copy.deepcopy(cad_list.detach().cpu().numpy())
    for i, cad in enumerate(cad_list_copy):
        cad[cad<-1] = -1
        cad[cad>1] = 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(cad))
        ipc = config.getint("n_cad_per_class", 1)
        name = get_class_name_from_id(i//ipc)
        o3d.io.write_point_cloud(config.get("save_path")+name+".ply", pcd)


class TensorDataset(Dataset):
    def __init__(self, cad, labels): 
        self.cad = cad.detach()
        self.labels = labels.detach()

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
