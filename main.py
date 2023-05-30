import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn
from utils.utils import get_loops, get_rand_cad, get_cad_points, get_time
import configparser
import os
import numpy as np
import torch.nn as nn


if __name__ == "__main__":
    print("=======Distillation========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("configs/distillation_config.ini")
    def_conf = config["DEFAULT"]
    ipc = def_conf.getint("n_cad_per_class", 1) ## Number of CAD models per class
    channel = def_conf.getint("channel", 1) ## Should it actually be a channel? 
    num_classes = def_conf.getint("num_classes", 2)
    outer_loop, inner_loop = get_loops(ipc) ## Variable defines the number of models per class
    if not os.path.exists(def_conf.get("save_path")):
        os.mkdir(def_conf.get("save_path"))

    batch_size = def_conf.getint("batch_size")
    if(def_conf.getboolean("overfit_1")): batch_size = 1

    grid_size = [40, 40, 40]
    
    train_set = ShapeNetPCD(
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="train"
    )
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size=batch_size,
        collate_fn=minkowski_collate_fn,
        shuffle=True,
        num_workers=def_conf.getint("num_workers", 2)
    )

    val_set = ShapeNetPCD(
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="val"
    )
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
        collate_fn=minkowski_collate_fn,
    )

    CAD_list, label_list = ShapeNetPCD(
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="train"
    ).load_data(data_root=def_conf.get("shapenet_path"), overfit_1=False, cls_name="airplane")
   

    indices_class = [[] for c in range(2)]
    for i, lab in enumerate(label_list):
        indices_class[lab].append(i)

    ## REAL initialization. 

    cad_syn = torch.randn(size=(num_classes*ipc, channel, grid_size[0], grid_size[1], grid_size[2]), requires_grad=True, device=device) ## This is a normal distribution from with mean 0, variance 1. 
    ## I can also divide by 2 and add 1 so it is from 0 to 1
    label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1)

    print(cad_syn.shape)
    print('initialize synthetic data from random real pcd')

    ### For REAL initialization
    for c in range(num_classes):
        path_to_rand_cad = get_rand_cad(c, ipc, indices_class, CAD_list)
        cad_pts = get_cad_points(path_to_rand_cad, def_conf.getint("num_points"))
        cad_syn.data[c*ipc:(c+1)*ipc] = cad_pts.detach().data
    ''' training '''
    optimizer_img = torch.optim.SGD([cad_syn, ], lr=def_conf.getfloat("lr_cad", 0.1), momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)

    print('%s training begins'%get_time())

    
