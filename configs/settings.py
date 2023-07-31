import argparse, configparser, logging, os, copy
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.ConvNet3D import ConvNet3D
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model
from utils.ClassShapeNet import ClassShapeNetDataset
import open3d as o3d
from utils.train_val_split import CLASS_NAME_TO_ID

def get_optimizer(model_params, target="dist", opt=""):
    if(opt==""):
        opt = modelconfig.get("dist_opt")
    opt = str.lower(opt)

    if(target == "dist"):
        lr = modelconfig.getfloat("dist_lr", 0.1)
    elif(target=="distnet"):
        lr = modelconfig.getfloat("distnet_lr", 0.1)
    else:
        lr = modelconfig.getfloat("classifier_lr", 0.1)


    if(opt=="sgd"):
        return torch.optim.SGD(
            model_params,
            lr=lr,
            momentum=modelconfig.getfloat("dist_momentum", 0.1)
        )
    else:
        return torch.optim.Adam(
            params=model_params,
            lr=lr,
            betas=[modelconfig.getfloat("dist_beta_1", 0.9), modelconfig.getfloat("dist_beta_2", 0.99)],
            eps=1e-08,
            weight_decay=1e-4,
        )
def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_network(name: str):
    name = str.lower(name)
    if(name=="pointnet" or name=="pointnet++"):
        return get_model(num_classes, False)
    elif(name=="convnet" or name=="convnet3d"):
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        net = ConvNet3D(channel=3, num_classes=num_classes, num_points=num_points, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
        return net

def log_string(str):
    if(DEBUG):
        logger.info(str)
    print(str)

def log_tensorboard(tag, scalar_value, global_step):
    if(DEBUG):
        summary_writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

def get_rand_cad_loader(c, n):
    cad_paths = get_rand_cad(c, n)
    dataset = ClassShapeNetDataset(cad_path=cad_paths, class_id=c)
    return DataLoader(dataset=dataset, batch_size=modelconfig.getint("batch_size", 4), shuffle=False)

def get_fixed_cad_loader(c):
    cad_paths = cad_paths_all[indices_class[c][:1]]
    dataset = ClassShapeNetDataset(cad_path=cad_paths, class_id=c)
    return DataLoader(dataset=dataset, batch_size=modelconfig.getint("batch_size", 4), shuffle=False)

def get_fixed_cad_path(c):
    cad_paths = cad_paths_all[indices_class[c][:1]]
    return cad_paths

def get_rand_cad(c, n):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return cad_paths_all[idx_shuffle]

def init_indices(n_classes):
    global indices_class
    indices_class = [[] for c in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

def set_labels_all(arr):
    global labels_all
    labels_all = copy.deepcopy(arr)

def set_cad_paths_all(arr):
    global cad_paths_all
    cad_paths_all = copy.deepcopy(arr)

def init():
    global device
    global DEBUG
    global config
    global defconfig
    global shapenetconfig
    global modelconfig
    global distillationconfig
    global mnistconfig
    global logger
    global summary_writer
    global num_classes
    global cad_paths_all
    global labels_all
    global indices_class
    global num_workers
    global exp_file_name
    global cad_per_class
    global num_points
    global batch_size

    cad_paths_all = []
    labels_all = []
    indices_class = []

    num_classes = 55

    now = datetime.now()
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log this trial? Default is True",
    )
    args = parser.parse_args()
    DEBUG = args.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("configs/default.ini")
    defconfig = config["DEFAULT"]
    shapenetconfig = config["SHAPENET"]
    modelconfig = config["MODEL"]
    distillationconfig = config["DISTILLATION"]
    mnistconfig = config["MNIST"]
    date_time = now.strftime("%Y%m%d%H%M%S")
    num_workers = defconfig.getint("num_workers")
    exp_file_name = date_time+"_"+defconfig.get("experiment_name")
    cad_per_class = distillationconfig.getint("cad_per_class")
    num_points = shapenetconfig.getint("num_points")
    batch_size = distillationconfig.getint("batch_size")
    logger = logging.getLogger("Distillation")
    logger.setLevel(logging.INFO)
    if(DEBUG):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging_file_path = os.path.join(defconfig.get("log_dir"),exp_file_name)
        print(logging_file_path)
        file_handler = logging.FileHandler(logging_file_path+".txt")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_string("INI Config Reading [DEFAULT] section:" + str(dict(defconfig)))
        tensorboard_path = os.path.join(defconfig.get("tensorboard_dir"),exp_file_name)
        summary_writer = SummaryWriter(log_dir=tensorboard_path)

def get_class_name_from_id(val):
    return [k for k, v in CLASS_NAME_TO_ID.items() if v == val][0]

def save_cad(cad_list, directory, iteration):
    cad_list_copy = cad_list.clone().detach().cpu().numpy()
    for i, cad in enumerate(cad_list_copy):
        cad[cad < -1] = -1
        cad[cad > 1] = 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(cad))
        ipc = cad_per_class
        name = get_class_name_from_id(i // ipc)
        o3d.io.write_point_cloud(
            directory
            + "/"
            + name
            + "_"
            + str(iteration)
            + ".ply",
            pcd,
        )