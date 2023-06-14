import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import open3d as o3d
import numpy as np
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
import MinkowskiEngine as ME
from tqdm import tqdm
from classification_model.train_val_split import TRAIN_DICT, VAL_DICT

class_id = {'pillow': 0, 'bowl': 1, 'rocket': 2, 'keyboard': 3, 'sofa': 4, 'car': 5, 'laptop': 6, 'jar': 7, 'chair': 8, 'rifle': 9, 'watercraft': 10, 'telephone': 11, 'bottle': 12, 'cellphone': 13, 'airplane': 14, 'bookshelf': 15, 'lamp': 16, 'bus': 17, 'birdhouse': 18, 'faucet': 19, 'table': 20, 'stove': 21, 'cap': 22, 'can': 23, 'mailbox': 24, 'bag': 25, 'loudspeaker': 26, 'piano': 27, 'knife': 28, 'guitar': 29, 'bench': 30, 'train': 31, 'display': 32, 'dishwasher': 33, 'microwaves': 34, 'bathtub': 35, 'helmet': 36, 'file cabinet': 37, 'trash bin': 38, 'cabinet': 39, 'motorbike': 40, 'flowerpot': 41, 'basket': 42, 'tower': 43, 'camera': 44, 'pistol': 45, 'remote': 46, 'skateboard': 47, 'printer': 48, 'bed': 49, 'mug': 50, 'washer': 51, 'microphone': 52, 'clock': 53, 'earphone': 54}

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"].reshape(1) for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

class ShapeNetPCD(Dataset):
    def __init__(
            self,
            phase: str,
            data_root: str,
            config,
            transform = None,
            num_points = 2048,
            for_distillation = False
        ) -> None:
        Dataset.__init__(self)
        classification_mode = config.get("classification_mode")
        cls_name = config.get("binary_class_name")
        self.phase = "val" if phase in ["val", "test"] else "train"
        self.data, self.label = self.load_data(data_root, classification_mode, cls_name, for_distillation)
        self.transform = transform
        self.num_points = num_points
        self.classification_mode = classification_mode

    def load_data(self, data_root, classification_mode, cls_name=None, for_distillation=False):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        if(cls_name is not None):
            target_class_dir = os.path.join(data_root,cls_name)

        if(self.phase == "train"):
            if(classification_mode == "multi"):
                for key in TRAIN_DICT.keys():
                    for model in TRAIN_DICT[key]:
                        labels.append(class_id[key])
                        data.append(model)
            else:
                for model in TRAIN_DICT[cls_name]:
                    labels.append(1)
                    data.append(model)
                for key in TRAIN_DICT.keys():
                    if key != cls_name:
                        for model in TRAIN_DICT[key][:80]:
                            labels.append(0)
                            data.append(model)

        if(self.phase =="val"):
            if(classification_mode == "multi"):
                for key in VAL_DICT.keys():
                    if(for_distillation):
                        for model in VAL_DICT[key][:20]:
                            labels.append(class_id[key])
                            data.append(model)
                    else:
                        for model in VAL_DICT[key]:
                            labels.append(class_id[key])
                            data.append(model)
            else:
                for model in VAL_DICT[cls_name]:
                    labels.append(1)
                    data.append(model)
                for key in VAL_DICT.keys():
                    if key != cls_name:
                        for model in VAL_DICT[key][:20]:
                            labels.append(0)
                            data.append(model)

        if(classification_mode == "overfit_1"):
            data = data[:1]
            labels = labels[:1]
        elif(classification_mode == "overfit_10"):
            data = data[:10]
            labels = labels[:10]

        labels = np.asarray(labels)
        
        print("Class counts ", np.unique(labels, return_counts=True))


        # if(overfit_1):
        #     for ply in tqdm(os.listdir(target_class_dir)[:2]):
        #         labels.append(1)
        #         data.append(os.path.join(target_class_dir,ply))
        #     return np.asarray(data),  torch.from_numpy(np.asarray(labels))

        # for ply in tqdm(os.listdir(target_class_dir)):
        #         labels.append(1)
        #         data.append(os.path.join(target_class_dir,ply))
        # for cls in os.listdir(data_root):
        #     if(cls == cls_name): continue
        #     files = os.path.join(data_root,cls)
        #     assert len(os.listdir(files)) > 0, "No files found"
        #     for ply in tqdm(os.listdir(files)[:100]):
        #         labels.append(0)
        #         data.append(os.path.join(files,ply))
        # labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        if(classification_mode == "multi"): labels.type(torch.LongTensor)
        return np.asarray(data),  labels
    
    def __getitem__(self, i):
        pcd = o3d.io.read_point_cloud(self.data[i])
        voxel_sz = 0.025
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_sz)
        # while(np.asarray(downpcd.points).shape[0] > self.num_points):
        #     voxel_sz += 0.05
        #     downpcd = pcd.voxel_down_sample(voxel_size=voxel_sz)

        xyz = np.asarray(downpcd.points)
        if self.phase == "train":
            np.random.shuffle(xyz)
            xyz = xyz[:self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = torch.from_numpy(xyz)
        return {
            "coordinates": xyz.to(torch.float32),
            "features": xyz.to(torch.float32),
            "label": label,
        }
    
    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return f"SHAPENET(phase={self.phase}, length={len(self)}, transform={self.transform})"

# def make_data_loader(phase, config):
#     assert phase in ["train", "val", "test"]
#     is_train = phase == "train"
#     dataset = ShapeNetPCD(
#         phase = phase,
#         transform=CoordinateTransformation(trans=float(config.get("train_translation")))
#         if is_train
#         else CoordinateTranslation(float(config.get("test_translation"))),
#         data_root=config.get("shapenet_path")
#     )

#     return DataLoader(
#         dataset=dataset,
#         num_workers=int(config.get("num_workers")),
        # shuffle=is_train,
        # collate_fn=minkowski_collate_fn,
        # batch_size=int(config.get("batch_size"))
#     )