import numpy as np
import torch
import time
import open3d as o3d


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
