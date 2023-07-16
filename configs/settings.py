import argparse, configparser, logging, os, copy
from datetime import datetime
import torch
import numpy as np

def log_string(str):
    if(DEBUG):
        logger.info(str)
    print(str)

def get_rand_cad(c, n):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return cad_paths_all[idx_shuffle]

def init_indices(n_classes):
    global indices_class
    num_classes = n_classes
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
    global logger
    global num_classes
    global cad_paths_all
    global labels_all
    global indices_class

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


    logger = logging.getLogger("Distillation")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    date_time = now.strftime("%Y%m%d%H%M%S")
    logging_file_path = os.path.join(defconfig.get("log_dir"),date_time+"_"+defconfig.get("experiment_name"))
    print(logging_file_path)
    file_handler = logging.FileHandler(logging_file_path+".txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("INI Config Reading [DEFAULT] section:" + str(dict(defconfig)))