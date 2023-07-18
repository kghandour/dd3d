import configs.settings as settings
import torch
import numpy as np
from utils.train_val_split import CLASS_NAME_TO_ID
from torch.utils.data import Dataset

def init_synth(
    cls_list = [],
):
    if(len(cls_list)>0):
        synthetic_xyz = torch.rand(
            size=(len(cls_list) * settings.cad_per_class, settings.num_points, 3), device=settings.device
        )
        synthetic_xyz = ((-1-1)*synthetic_xyz + 1).requires_grad_()
        synthetic_label = torch.tensor(np.array([np.ones(settings.cad_per_class, dtype=int)*CLASS_NAME_TO_ID[i] for i in cls_list]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1)
        return synthetic_xyz, synthetic_label
    
    synthetic_xyz = torch.rand(
        size=(settings.num_classes * settings.cad_per_class, settings.num_points, 3), device=settings.device
    )
    synthetic_xyz = ((-1-1)*synthetic_xyz + 1).requires_grad_()
    synthetic_label = torch.tensor(np.array([np.ones(settings.cad_per_class, dtype=int)*i for i in range(settings.num_classes)]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1)
    return synthetic_xyz, synthetic_label

    
class SyntheticDataset(Dataset):
    def __init__(
            self,
            xyz_list,
            labels_list
        ):
        self.xyz_list = xyz_list
        self.labels_list = labels_list

    def __getitem__(self, index):
        return self.xyz_list[index], self.labels_list[index]
    
    def __len__(self):
        return self.xyz_list.shape[0]
