from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import torch

class ClassShapeNetDataset(Dataset):
    def __init__(self, cad_path = [], class_id=-1):
        assert len(cad_path)>0, f'Incorrect usage of ClassShapeNetDataset. Must pass a list of cad'
        assert class_id > -1, f'Incorrect usage of ClassShapeNetDataset. Must pass class id > -1'

        self.data = cad_path
        self.class_id = class_id

    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.data[index])
        xyz = np.asarray(pcd.points)
        np.random.shuffle(xyz)
        xyz = xyz[:self.num_points]
        xyz = torch.from_numpy(xyz)
        return xyz, self.labels[index]
    
    def __len__(self):
        return self.data.shape[0]