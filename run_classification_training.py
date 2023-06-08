import configparser
import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.me_network import MinkowskiFCNN
from classification_model.me_classification import train, test
from torch.utils.tensorboard import SummaryWriter
import time
import os


from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("configs/classification_load.ini")
    def_conf = config["DEFAULT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_mode = def_conf.get("classification_mode")
    target_class = def_conf.get("binary_class_name")
    exp_name = classification_mode+"_"+def_conf.get("exp_name")+"_"+target_class+"_"+str(time.time())
    if(classification_mode=="multi"): exp_name = classification_mode+"_"+def_conf.get("exp_name")+"_"+str(time.time())
    load_model_path = def_conf.get("load_model")
    load_model = False
    if(load_model_path is not None):
        load_model = True
        print("========LOADING SAVED WEIGHTS========")

    print("Classification mode: ", classification_mode)

    print("=============ShapeNet PCD Dataset===============")
    print(f"Training with translation", def_conf.get("train_translation"))
    print(f"Evaluating with translation", def_conf.get("train_translation"))
    print("=============================================\n\n")


    if(classification_mode=="multi"):
        net = MinkowskiFCNN(
            in_channel=3, out_channel=55, embedding_channel=1024, classification_mode = classification_mode
        ).to(device)
    else:
        net = MinkowskiFCNN(
            in_channel=3, out_channel=1, embedding_channel=1024, classification_mode = classification_mode
        ).to(device)


    print("===================Network===================")
    print(net)
    print("=============================================\n\n")
    
    # dataset = ShapeNetPCD(
    #     transform=CoordinateTransformation(trans=float(def_conf.get("train_translation"))),
    #     data_root=def_conf.get("shapenet_path"),
    #     config = def_conf
    # )

    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(float(def_conf.get("validation_split")) * dataset_size))
    # if(overfit_1): split = int(np.floor(0.5 * dataset_size))
    # if bool(def_conf.get("shuffle_dataset")) :
    #     np.random.seed(int(def_conf.get("random_seed")))
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    batch_size = int(def_conf.get("batch_size"))
    if(classification_mode == "overfit_1"): batch_size = 1

    if(load_model):
        loaded_dict = torch.load(load_model_path)
        net.load_state_dict(loaded_dict['state_dict'])
        net.eval()

    train_set = ShapeNetPCD(
        transform=CoordinateTransformation(trans=float(def_conf.get("train_translation"))),
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="train"

    )
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size=batch_size,
        collate_fn=minkowski_collate_fn,
        shuffle=True,
        drop_last = True
    )

    val_set = ShapeNetPCD(
        transform=CoordinateTranslation(float(def_conf.get("test_translation"))),
        data_root=def_conf.get("shapenet_path"),
        config=def_conf,
        phase="val"
    )
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
        collate_fn=minkowski_collate_fn,
        drop_last = True
    )

    print("==================Init Logger===============\n\n")

    writer = SummaryWriter(log_dir=os.path.join(def_conf.get("log_dir"),exp_name)) #initialize sumamry writer
    print("Initialized to ", os.path.join(def_conf.get("log_dir"),exp_name))

    train(net, device, def_conf, writer, train_dataloader=train_loader, val_loader=validation_loader)
    print("====== TRAINING COMPLETE =======")
    print("====== TESTING MODE ==========")
    accuracy = test(net, device, def_conf, phase="test", val_loader=validation_loader)
    print()
    print(f"Test accuracy: {accuracy}")
    

