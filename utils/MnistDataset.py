from torch.utils.data import Dataset
import configs.settings as settings
import os, h5py
import numpy as np
import torch

class MnistDataset(Dataset):
    def __init__(
            self,
            phase="train"
        ):
        super().__init__()
        self.data, self.labels = self.load_data(phase)
    
    def load_data(self, phase):
        data, labels = [], []
        data_dir = settings.mnistconfig.get("dataset_dir")
        assert os.path.exists(data_dir), f"{data_dir} does not exist. Please set it in the default.ini under [MNIST]"
        if(phase == "train"):
            data_path = os.path.join(data_dir, "train_point_clouds.h5")
        elif(phase == "val" or phase == "test"):
            data_path = os.path.join(data_dir, "test_point_clouds.h5")
        else:
            exit("Unknown phase. Please specify either train or val or test")
        with h5py.File(data_path) as f:
            for key in f.keys():
                data.append(f[key]["points"][:].astype("float32"))
                labels.append(f[key].attrs['label'].astype("int64"))
        labels = np.stack(labels, axis=0)
        return data, labels
    
    def __getitem__(self, index):
        xyz = self.data[index]
        xyz = np.array(xyz)
        np.random.shuffle(xyz)
        if(len(xyz)>settings.num_points):
            xyz = xyz[: settings.num_points]
        label = self.labels[index]
        xyz = torch.from_numpy(xyz)
        label = torch.from_numpy(xyz)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label
        }
    
    def __len__(self):
        return self.data.shape[0]
