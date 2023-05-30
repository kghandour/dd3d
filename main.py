import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn
from utils.utils import get_loops
import configparser
import os
import numpy as np

if __name__ == "__main__":
    print("=======Distillation========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_loop, inner_loop = get_loops(1) ## Variable defines the number of models per class
    config = configparser.ConfigParser()
    config.read("configs/distillation_config.ini")
    def_conf = config["DEFAULT"]
    num_classes = 2
    if not os.path.exists(def_conf.get("save_path")):
        os.mkdir(def_conf.get("save_path"))

    batch_size = def_conf.getint("batch_size")
    if(def_conf.getboolean("overfit_1")): batch_size = 1
    
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

    model_ds, label_ds = ShapeNetPCD(
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="train"
    ).load_data(data_root=def_conf.get("shapenet_path"), overfit_1=False, cls_name="airplane")
   

    indices_class = [[] for c in range(2)]
    for i, lab in enumerate(label_ds):
        indices_class[lab].append(i)

    
    
