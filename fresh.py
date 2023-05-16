from pytorch3d.datasets import ShapeNetCore
import configparser
import torch
from model.augmentation import CoordinateTransformation
from model.me_network import MinkowskiFCNN
from model.me_classification import train, test
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


from model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    def_conf = config["DEFAULT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overfit_1 = bool(def_conf.getboolean("overfit_1"))
    print("=====OVERFITTING?====", overfit_1)

    print("===================ModelNet40 Dataset===================")
    print(f"Training with translation", def_conf.get("train_translation"))
    print(f"Evaluating with translation", def_conf.get("train_translation"))
    print("=============================================\n\n")

    net = MinkowskiFCNN(
        in_channel=3, out_channel=2, embedding_channel=1024, overfit_1=overfit_1
    ).to(device)

    print("===================Network===================")
    print(net)
    print("=============================================\n\n")
    print("==================Init Logger===============\n\n")
    writer = SummaryWriter(log_dir=os.path.join(def_conf.get("log_dir"),def_conf.get("exp_name")+str(time.time()))) #initialize sumamry writer
    print("Initialized to ", os.path.join(def_conf.get("log_dir"),def_conf.get("exp_name")+str(time.time())))
    dataset = ShapeNetPCD(
        transform=CoordinateTransformation(trans=float(def_conf.get("train_translation"))),
        data_root=def_conf.get("shapenet_path"),
        config = def_conf
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(float(def_conf.get("validation_split")) * dataset_size))
    if(overfit_1): split = int(np.floor(0.5 * dataset_size))
    if bool(def_conf.get("shuffle_dataset")) :
        np.random.seed(int(def_conf.get("random_seed")))
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = int(def_conf.get("batch_size"))
    if(overfit_1): batch_size = 1

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        sampler=train_sampler, 
        collate_fn=minkowski_collate_fn,
        )
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=minkowski_collate_fn,)

    train(net, device, def_conf, writer, train_dataloader=train_loader, val_loader=validation_loader, overfit_1=overfit_1)

    accuracy = test(net, device, def_conf, phase="test")
    print(f"Test accuracy: {accuracy}")
    

