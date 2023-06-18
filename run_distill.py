from sklearn import metrics
import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn
from utils.utils import RealTensorDataset, TensorDataset, get_loops, get_rand_cad, get_cad_points, get_time, match_loss, save_cad, list_average
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



def evaluate_synset(it, iteration, net, syn_ds, val_loader, config, checkpoint_load, device):
    optimizer_model = optim.SGD(
        net.parameters(),
        lr=config.getfloat("lr_classification"),
        momentum=0.9,
        weight_decay=config.getfloat("weight_decay_classification"),
    )
    scheduler_model = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model,
        T_max=config.getint("max_steps"),
    )
    optimizer_model.load_state_dict(checkpoint_load['optimizer'])
    scheduler_model.load_state_dict(checkpoint_load['scheduler'])
    time_start = time.time()

    epoch_eval_train = config.getint("epoch_eval_train")
    loss_train, acc_train = 0, 0
    ## TODO: Initial randomly init synth heavily modifies the classifier network. Reduced Epochs to 1 from 10 to reduce impact.
    for ep in range(epoch_eval_train+1):
        loss_train, acc_train = train_classifier(net, device, config, syn_ds, "train", optimizer_model, scheduler_model)
    time_train = time.time() - time_start
    loss_val, acc_val = train_classifier(net, device, config, val_loader, "val", optimizer_model, scheduler_model)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, epoch_eval_train, int(time_train), loss_train, acc_train, acc_val))
    return acc_train, acc_val, time_train, loss_train, loss_val




def train_classifier(net, device, config, loader, phase, optimizer, scheduler):
    if(phase=="train"):
        net.train()
    else:
        net.eval()
    
    loss_avg, acc_avg, num_exp = 0, 0, 0

    for train_iter in loader:
        input = create_input_batch(
            train_iter, device=device, quantization_size=config.getfloat("voxel_size")
        )
        # print("Input batch shape for classifier ", input.shape)
        logit = net(input)
        # print("Network Classifier output shape ", logit.shape, "Label output shape", train_iter["labels"].shape)
        loss = criterion(logit, train_iter["labels"].to(device), config.get("classification_mode"))
        accuracy = metrics.accuracy_score(train_iter["labels"].cpu(), torch.argmax(logit, 1).cpu())

        if(phase=="train"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        torch.cuda.empty_cache()

        
        loss_avg += loss.item()
        acc_avg += accuracy
        num_exp += 1

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

if __name__ == "__main__":
    print("=======Distillation========")
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True, help="Log this trial? Default is True")
    args = parser.parse_args()
    logging = args.log
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("configs/distillation_config.ini")
    def_conf = config["DEFAULT"]
    ipc = def_conf.getint("n_cad_per_class", 1) ## Number of CAD models per class
    channel = def_conf.getint("channel", 3) ## Should it actually be a channel? 
    num_classes = def_conf.getint("num_classes", 2)
    outer_loop, inner_loop = get_loops(ipc) ## Variable defines the number of models per class
    total_iterations = def_conf.getint("total_iterations")
    eval_iteration_pool = np.arange(0, total_iterations+1, def_conf.getint("save_cad_and_eval_every")).tolist()
    num_points = def_conf.getint("num_points", 2048) ## Number of points
    load_model_path = def_conf.get("load_model")
    loaded_dict = torch.load(load_model_path)
    if not os.path.exists(def_conf.get("save_path")):
        os.mkdir(def_conf.get("save_path"))

    batch_size = def_conf.getint("batch_size")
    if(def_conf.getboolean("overfit_1")): batch_size = 1

    
    ''' organize the real dataset '''
    cad_all_path = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    '''This loads the list of paths and labels of the trainset. 
    Does not make a val set because it is not needed atm.
    DS Stores the entire images in memory. Not the most efficient in terms of memory but faster'''
    cad_all_path, labels_all = ShapeNetPCD(phase="train", data_root=def_conf.get("shapenet_path"), config=def_conf).load_data(data_root=def_conf.get("shapenet_path"), classification_mode=def_conf.get("classification_mode"))
    val_set = ShapeNetPCD(phase="val", data_root=def_conf.get("shapenet_path"), config=def_conf, for_distillation=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True)
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    ## REAL initialization. 

    # RANDN results in a normal distribution. Values not limited from -1, 1.
    # cad_syn = torch.randn(size=(num_classes*ipc, num_points, channel), requires_grad=True, device=device) ## This is a normal distribution from with mean 0, variance 1. 
    # RAND might be a better choice (should be from 0 to 1 now)
    cad_syn = torch.rand(size=(num_classes*ipc, num_points, channel), device=device)
    ## Set it from -1 to 1
    cad_syn = ((-1-1)*cad_syn + 1).requires_grad_()

    # cad_syn_orig = cad_syn.clone().detach()

    label_syn = torch.tensor(np.array([np.ones(ipc, dtype=int)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1)


    ### For REAL initialization
    # if(def_conf.get("initialization")=="real"):
    #     print("======= Initializing synthetic dataset from real data ==========")
    #     for c in range(num_classes):
    #         path_to_rand_cad = get_rand_cad(c, ipc, indices_class, cad_all_path)
    #         cad_pts = get_cad_points(path_to_rand_cad, def_conf.getint("num_points"))
    #         cad_syn.data[c*ipc:(c+1)*ipc] = cad_pts.detach().data
    # else:
    #     print(f"======= Initialized Synthetic dataset with {ipc} CAD per class. CAD array shape is {cad_syn.shape} with values normalized between {cad_syn.min(), cad_syn.max()}===============")

    print("========= Initializing SummaryWriter ==========")
    if(logging):
        summary_writer = SummaryWriter(log_dir=os.path.join(def_conf.get("log_dir"),"distillation"+str(time.time()))) #initialize sumamry writer
    else: summary_writer=None

    ''' training '''
    optimizer_distillation = torch.optim.SGD([cad_syn, ], lr=def_conf.getfloat("lr_cad", 0.1), momentum=0.5) # optimizer_img for synthetic data
    optimizer_distillation.zero_grad()
    # criterion = nn.CrossEntropyLoss().to(device)

    model_eval_pool = ["MINKENGINE"]

    print('%s training begins'%get_time())
    for it in range(total_iterations):
        if it in eval_iteration_pool:
            log_acc_train, log_acc_test, log_time_train, log_loss_train, log_loss_val = [], [], [], [], []
            ## TODO find other possible evaluation models
            for model_eval in model_eval_pool:
                print("====== Testing Classifier =========")
                net_classifier = MinkowskiFCNN(
                    in_channel=3, out_channel=num_classes, embedding_channel=1024, classification_mode=def_conf.get("classification_mode")
                ).to(device)
                net_classifier.load_state_dict(loaded_dict['state_dict'])
                accs_train = []
                ## TODO: Check for the num_evals good value.
                for it_eval in range(def_conf.getint("num_eval")):
                    # cad_syn_eval, label_syn_eval = copy.deepcopy(cad_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    cad_syn_eval, label_syn_eval = cad_syn.clone().detach(), label_syn.clone().detach() # avoid any unaware modification
                    syn_ds = TensorDataset(cad_syn_eval, label_syn_eval)
                    syn_loader = torch.utils.data.DataLoader(syn_ds, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True)
                    acc_train, acc_test, time_train, loss_train, loss_val = evaluate_synset(it, it_eval, net_classifier, syn_loader, val_loader, def_conf, loaded_dict, device)
                    accs_train.append(acc_train)
                    log_acc_train.append(acc_train)
                    log_acc_test.append(acc_test)
                    log_time_train.append(time_train)
                    log_loss_train.append(loss_train)
                    log_loss_val.append(loss_val)
            if(logging):
                summary_writer.add_scalar("Classifier/Train_Accuracy", list_average(log_acc_train), it)
                summary_writer.add_scalar("Classifier/Train_Time", list_average(log_time_train), it)
                summary_writer.add_scalar("Classifier/Train_Loss", list_average(log_loss_train), it)
                summary_writer.add_scalar("Classifier/Val_Accuracy", list_average(log_acc_test), it)
                summary_writer.add_scalar("Classifier/Val_Loss", list_average(log_loss_val), it)
            log_acc_train, log_acc_test, log_time_train, log_loss_train, log_loss_val = [], [], [], [], []




        '''Save point cloud'''
        if((it + 1) % def_conf.getint("save_cad_and_eval_every") == 0):
            print("====== Exporting Point Clouds ======")
            save_cad(cad_syn, def_conf, it)


        # net_distillation = MinkowskiFCNN(in_channel=3, out_channel=num_classes, embedding_channel=1024, classification_mode=def_conf.get("overfit_1")).to(device)
        net_distillation = MinkowskiDistill(in_channel=3, out_channel=num_classes, embedding_channel=256, num_points=num_points).to(device)
        net_distillation.train()
        net_parameters = list(net_distillation.parameters())
    
        optimizer_dist_net = optim.SGD(
            net_distillation.parameters(),
            lr=def_conf.getfloat("lr_cad"),
            momentum=0.9,
            weight_decay=def_conf.getfloat("weight_decay_distillation"),
        )
        scheduler_dist_net = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_dist_net,
            T_max=def_conf.getint("max_steps"),
        )
        optimizer_dist_net.zero_grad()
        loss_avg = 0

        for ol in range(outer_loop):
            loss = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                cad_real_class = get_rand_cad(c, 4, indices_class, cad_all_path)
                lab_real_class = torch.ones((cad_real_class.shape[0],), device=device, dtype=torch.long) * c
                cad_syn_class = cad_syn[c*ipc:(c+1)*ipc].clone()
                lab_syn_class = torch.ones((ipc), device=device, dtype=torch.long) * c
                ## Shapes match expectation so far.
                ## TODO Create loader from array for real and synthetic
                input_real_ds = RealTensorDataset(cad_real_class, lab_real_class)
                input_real_loader = torch.utils.data.DataLoader(input_real_ds, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True)
                loss_real_list = []
                loss_syn_list = []
                for input_real_iter in input_real_loader:
                    input_real = create_input_batch(
                        input_real_iter, device=device, quantization_size=def_conf.getfloat("voxel_size")
                    )
                    output_real = net_distillation(input_real)
                    loss_real =  F.cross_entropy(output_real, input_real_iter["labels"].to(device), reduction="mean")
                    loss_real_list.append(loss_real)
                loss_real = sum(loss_real_list)/len(loss_real_list)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                syn_ds = TensorDataset(cad_syn_class, lab_syn_class)
                syn_loader = torch.utils.data.DataLoader(syn_ds, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=False)
                for input_real_iter in syn_loader:
                    input_real = create_input_batch(
                        input_real_iter, device=device, quantization_size=def_conf.getfloat("voxel_size")
                    )
                    output_syn = net_distillation(input_real)
                    loss_syn =  F.cross_entropy(output_syn, input_real_iter["labels"].to(device), reduction="mean")
                    loss_syn_list.append(loss_syn)                
                loss_syn = sum(loss_syn_list)/len(loss_syn_list)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, "mse", device=device)

            optimizer_distillation.zero_grad()
            loss.backward()
            optimizer_distillation.step()
            loss_avg += loss.item()
            # if ol == outer_loop - 1:
            #     break
            # cad_syn_train, label_syn_train = copy.deepcopy(cad_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            cad_syn_train, label_syn_train = cad_syn.clone().detach(), label_syn.clone().detach() # avoid any unaware modification
            syn_ds = TensorDataset(cad_syn_train, label_syn_train)
            syn_loader = torch.utils.data.DataLoader(syn_ds, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True)
            for il in range(inner_loop):
                train_classifier(net_distillation, device, def_conf, syn_loader, "train", optimizer_dist_net, scheduler_dist_net)
        
        loss_avg /= (num_classes*outer_loop)

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))
            if(logging): summary_writer.add_scalar("Distillation/Loss_Average", loss_avg, it)
        if( it%100 == 0):
            print("=== Saving Model =====")
            torch.save(
                {
                    "dist_state": net_distillation.state_dict(),
                    "optimizer_dist_net": optimizer_dist_net.state_dict(),
                    "optimizer_distillation": optimizer_distillation.state_dict(),
                    "scheduler": scheduler_dist_net.state_dict(),
                    "curr_iter": it,
                    "cad_syn":cad_syn.clone(),
                    "label_syn":label_syn.clone()
                },
                def_conf.get("distillation_model_name"),
            )


        