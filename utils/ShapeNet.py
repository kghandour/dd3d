from torch.utils.data import Dataset
import os, torch
import configs.settings as settings
from utils.train_val_split import TRAIN_DICT, VAL_DICT, CLASS_NAME_TO_ID, MODELNET_NAME_TO_ID
import numpy as np
import configs.settings as settings
import open3d as o3d
import copy

class ShapeNetDataset(Dataset):
    def __init__(self, cls_list=[], phase="train", modelnet40=False):
        super().__init__()
        self.phase = "val" if phase in ["val", "test"] else "train"
        self.num_points = settings.shapenetconfig.getint("num_points", 2048)
        self.cls_list = cls_list
        self.dict_to_id = CLASS_NAME_TO_ID
        if(modelnet40):
            self.dict_to_id = MODELNET_NAME_TO_ID     
        self.data, self.labels = self.load_data()
        settings.set_cad_paths_all(self.data)
        settings.set_labels_all(self.labels)
        if(modelnet40):
            n_class = 40
        else: n_class = self.labels.unique().shape[0]
        settings.init_indices(n_class)
  

    def load_data(self):
        data, labels = [], []
        dataset_path = settings.shapenetconfig.get("dataset_dir")
        assert os.path.exists(dataset_path), f"{dataset_path} does not exist."
        if(len(self.cls_list)>0):
            for class_name in self.cls_list:
                if(class_name not in self.dict_to_id):
                    continue
                if(self.phase == "train"):
                    for model in TRAIN_DICT[class_name]:
                        labels.append(self.dict_to_id[class_name])
                        data.append(model)
                else:
                    for model in VAL_DICT[class_name]:
                        labels.append(self.dict_to_id[class_name])
                        data.append(model)
                    
            return np.asarray(data), torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
        
        if(self.phase == "train"):
            for key in TRAIN_DICT.keys():
                if(key not in self.dict_to_id):
                    continue
                for model in TRAIN_DICT[key]:
                    labels.append(self.dict_to_id[key])
                    data.append(model)

        else:
            for key in VAL_DICT.keys():
                if(key not in self.dict_to_id):
                    continue
                for model in VAL_DICT[key]:
                    labels.append(self.dict_to_id[key])
                    data.append(model)
        
        settings.log_string("Class counts " +str(np.unique(labels, return_counts=True)[1]))
        return np.asarray(data), torch.from_numpy(np.asarray(labels)).type(torch.LongTensor)
    
    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.data[index])
        xyz = np.asarray(pcd.points)
        np.random.shuffle(xyz)
        xyz = xyz[:self.num_points]
        xyz = torch.from_numpy(xyz).type(torch.float32)
        return xyz, self.labels[index]
    
    def __len__(self):
        return self.data.shape[0]