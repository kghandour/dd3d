from sklearn import metrics
import torch
from tqdm import tqdm
from classification_model.augmentation import (
    CoordinateTransformation,
    CoordinateTranslation,
)
from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn
from distillation_model.classification_block import (
    classification_evaluation_block,
    run_epoch,
)
from distillation_model.dist_blocks import outer_block
from utils.utils import (
    RealTensorDataset,
    create_loader_for_synthetic_cad,
    create_val_loader_and_list,
    get_loops,
    get_rand_cad,
    get_cad_points,
    get_real_cad_paths_and_labels,
    get_time,
    initalize_synthetic_tensors,
    log_classification_metrics_and_reset,
    match_loss,
    populate_classification_metrics_dict,
    save_cad,
    list_average,
)
from distillation_model.resnet import ResNet14, ResNet18
import configparser
import os
import numpy as np
import torch.nn as nn
from classification_model.me_network import MinkowskiFCNN, criterion
from classification_model.me_classification import create_input_batch, test
import copy
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from distillation_model.dist_network import MinkowskiDistill
import torch.nn.functional as F

if __name__ == "__main__":
    print("=======Distillation========")
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log this trial? Default is True",
    )
    args = parser.parse_args()
    logging = args.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("configs/distillation_config.ini")
    def_conf = config["DEFAULT"]
    ipc = def_conf.getint("n_cad_per_class", 1)  ## Number of CAD models per class
    num_classes = def_conf.getint("num_classes", 2)
    outer_loop, inner_loop = get_loops(
        ipc
    )  ## Variable defines the number of models per class
    total_iterations = def_conf.getint("total_iterations")
    eval_iteration_pool = [total_iterations-1]
    num_points = def_conf.getint("num_points", 2048)  ## Number of points
    load_classification_model_path = def_conf.get("load_model")
    loaded_classification_dict = torch.load(load_classification_model_path)
    classes_to_distill = []

    distillation_out_path = os.path.join(
        def_conf.get("save_path"), def_conf.get("distillation_exp_name")
    )
    if not os.path.exists(distillation_out_path):
        os.mkdir(distillation_out_path)

    batch_size = def_conf.getint("batch_size", 4)
    if def_conf.getboolean("overfit_1"):
        batch_size = 1

    ## We need a list of all cad paths and a val_loader
    cad_all_path, labels_all = [], []
    cad_all_path, labels_all, val_loader = create_val_loader_and_list(
        def_conf=def_conf, classes_to_distill=classes_to_distill
    )

    ## 2D array where index 0 is the class ID, and index 1 is an array of labels in that list
    indices_class = [[] for c in range(num_classes)]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    ## Debugging Purposes
    for c in range(num_classes):
        print("class c = %d: %d real images" % (c, len(indices_class[c])))

    ## Initialize synthetic tensors
    cad_syn_tensor, label_syn_tensor = initalize_synthetic_tensors(
        num_classes=num_classes,
        ipc=ipc,
        device=device,
        num_points=num_points,
        initialization="random",
        classes_to_distill=classes_to_distill,
    )

    ## Checks if we want to initialize a logger or not
    if logging:
        print("========= Initializing SummaryWriter ==========")
        summary_writer = SummaryWriter(
            log_dir=os.path.join(
                def_conf.get("log_dir"),
                "distillation_"
                + def_conf.get("distillation_exp_name")
                + str(time.time()),
            )
        )  # initialize sumamry writer
    else:
        summary_writer = None

    ## Initializing Optimizer for the synthetic CAD.
    ## I tried SGD. maybe try ADAM
    # optimizer_distillation = torch.optim.Adam(
    #     [
    #         cad_syn_tensor,
    #     ],
    #     lr=def_conf.getfloat("lr_cad", 0.1),
    #     betas=[0.9, 0.99],
    # )

    optimizer_distillation = torch.optim.SGD(
        [
            cad_syn_tensor,
        ],
        lr=def_conf.getfloat("lr_cad", 0.1),
    )
    optimizer_distillation.zero_grad()
    model_eval_pool = ["MINKENGINE"]

    ## Initialize and load weights for PRETRAINED classification network
    net_classification = MinkowskiFCNN(
        in_channel=3,
        out_channel=num_classes,
        embedding_channel=1024,
        classification_mode=def_conf.get("classification_mode"),
    ).to(device)
    net_classification.load_state_dict(loaded_classification_dict["state_dict"])
    optimizer_classification = optim.SGD(
        net_classification.parameters(),
        lr=def_conf.getfloat("lr_classification"),
        momentum=0.9,
        weight_decay=def_conf.getfloat("weight_decay_classification"),
    )
    scheduler_classification = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_classification,
        T_max=def_conf.getint("max_steps"),
    )
    optimizer_classification.load_state_dict(loaded_classification_dict["optimizer"])
    scheduler_classification.load_state_dict(loaded_classification_dict["scheduler"])

    ## Initialize and load weights for PRETRAINED DISTILLATION NETWORK
    # net_distillation = MinkowskiFCNN(
    #     in_channel=3,
    #     out_channel=num_classes,
    #     embedding_channel=1024,
    #     classification_mode=def_conf.get("classification_mode"),
    # ).to(device)
    # net_distillation.load_state_dict(loaded_classification_dict["state_dict"])
    # net_distillation.eval()

    # net_distillation = MinkowskiDistill(
    #     in_channel=3,
    #     out_channel=55,
    # ).to(device)

    net_distillation = ResNet18(
        in_channels=3,
        out_channels=55
    ).to(device)

    print(net_distillation)

    print("%s training begins" % get_time())

    ## Start Loop Here
    for it in tqdm(range(total_iterations), desc="Iterations: "):
        ## Classification Evaluation Blocks compares the classifier performance on the synthetic and validation set
        classification_evaluation_block(
            iteration=it,
            eval_iteration_pool=eval_iteration_pool,
            model_eval_pool=model_eval_pool,
            net_classification=net_classification,
            optimizer_classification=optimizer_classification,
            scheduler_classification=scheduler_classification,
            def_conf=def_conf,
            cad_syn_tensor=cad_syn_tensor,
            label_syn_tensor=label_syn_tensor,
            val_loader=val_loader,
            logging=logging,
            summary_writer=summary_writer,
            device=device,
            batch_size=batch_size,
        )

        """Save point cloud"""
        if (it % def_conf.getint("save_cad_and_eval_every")) == 0:
            print("====== Exporting Point Clouds ======")
            save_cad(
                cad_syn_tensor, def_conf, directory=distillation_out_path, iteration=it
            )

        ## END REFACTOR phase 1
        ## Start Refactor phase 2
        net_parameters = list(net_distillation.parameters())

        optimizer_dist_net = optim.SGD(
            net_distillation.parameters(),
            lr=def_conf.getfloat("lr_cad"),
            momentum=0.9,
            weight_decay=def_conf.getfloat("weight_decay_distillation"),
        )
        optimizer_dist_net.zero_grad()
        scheduler_dist_net = None

        loss_avg = 0
        dist_net_loss_list, dist_net_acc_list = [], []
        for ol in range(outer_loop):
            loss_avg += outer_block(
                num_classes=num_classes,
                indices_class=indices_class,
                cad_all_path=cad_all_path,
                device=device,
                cad_syn_tensor=cad_syn_tensor,
                label_syn_tensor=label_syn_tensor,
                ipc=ipc,
                def_conf=def_conf,
                distillation_network=net_distillation,
                net_parameters=net_parameters,
                optimizer_distillation=optimizer_distillation,
                classes_to_distill=classes_to_distill,
            )
            if ol == outer_loop - 1:
                break
            ## END REFACTOR PHASE 2
            syn_loader = create_loader_for_synthetic_cad(
                cad_syn_tensor=cad_syn_tensor, label_syn_tensor=label_syn_tensor
            )
            dist_net_loss_list, dist_net_acc_list = [], []
            for il in range(inner_loop):
                distilliation_net_loss, distilliation_net_acc = run_epoch(
                    network=net_distillation,
                    optimizer=optimizer_dist_net,
                    scheduler=scheduler_dist_net,
                    phase="train",
                    dataloader=syn_loader,
                    config=def_conf,
                    device=device,
                )
                dist_net_loss_list.append(distilliation_net_loss)
                dist_net_acc_list.append(distilliation_net_acc)
        if(logging and len(dist_net_loss_list)> 0):
            summary_writer.add_scalar("Distillation Network/Loss", list_average(dist_net_loss_list), it)
            summary_writer.add_scalar("Distillation Network/Accuracy", list_average(dist_net_acc_list), it)
        loss_avg /= num_classes * outer_loop

        if it % 10 == 0:
            print("%s iter = %04d, loss = %.4f" % (get_time(), it, loss_avg))
            if logging:
                summary_writer.add_scalar("Distillation/Loss_Average", loss_avg, it)
        if it % 100 == 0:
            print("=== Saving Model =====")
            torch.save(
                {
                    "dist_state": net_distillation.state_dict(),
                    "optimizer_dist_net": optimizer_dist_net.state_dict(),
                    "optimizer_distillation": optimizer_distillation.state_dict(),
                    "curr_iter": it,
                    "cad_syn": cad_syn_tensor.clone(),
                    "label_syn": label_syn_tensor.clone(),
                },
                def_conf.get("distillation_model_name")
                + def_conf.get("distillation_exp_name")
                + ".model",
            )
