import configparser
import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.me_network import MinkowskiFCNN
from classification_model.me_classification import train, test
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from classification_model.shapepcd_single import ShapeNetPCD_Single, minkowski_collate_fn
import torch.optim as optim


if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("configs/classification_load.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def_conf = config["DEFAULT"]
    classification_mode = def_conf.get("classification_mode")
    load_model_path = def_conf.get("load_model")

    
    ## Init model
    net = MinkowskiFCNN(
        in_channel=3, out_channel=55, embedding_channel=1024, classification_mode = classification_mode
    ).to(device)
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")


    ## Load weights
    print("========LOADING WEIGHTS=========")
    loaded_dict = torch.load(load_model_path)
    net.load_state_dict(loaded_dict['state_dict'])
    net.eval()

    item_set = ShapeNetPCD_Single("val", item_path=def_conf.get("test_ply_path"), transform=CoordinateTranslation(float(def_conf.get("test_translation"))),config=def_conf)
    validation_loader = torch.utils.data.DataLoader(item_set, batch_size=1,
        collate_fn=minkowski_collate_fn,
        drop_last = True
    )

    print("====== Init optimizer and scheduler ======")
    optimizer = optim.SGD(
        net.parameters(),
        lr=def_conf.getfloat("lr"),
        momentum=0.9,
        weight_decay=def_conf.getfloat("weight_decay"),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=def_conf.getint("max_steps"),
    )
    optimizer.load_state_dict(loaded_dict['optimizer'])
    scheduler.load_state_dict(loaded_dict['scheduler'])

    print("====== TESTING MODE ==========")
    accuracy = test(net, device, def_conf, phase="test", val_loader=validation_loader)
    print()
    print(f"Test accuracy: {accuracy}")





