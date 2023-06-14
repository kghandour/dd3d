from sklearn import metrics
import torch
from classification_model.augmentation import CoordinateTransformation, CoordinateTranslation
from classification_model.shapepcd_set import ShapeNetPCD, minkowski_collate_fn
from utils.utils import TensorDataset, get_loops, get_rand_cad, get_cad_points, get_time
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



def evaluate_synset(it, iteration, net, syn_ds, val_loader, config, checkpoint_load, device, summary_writer):
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
    ## TODO continue epoch passing through the synthetic, and through the original
    time_start = time.time()

    epoch_eval_train = config.getint("epoch_eval_train")
    loss_train, acc_train = 0, 0
    for ep in range(epoch_eval_train+1):
        loss_train, acc_train = train_classifier(net, device, config, syn_ds, "train", optimizer_model, scheduler_model)
        if(logging):
            summary_writer.add_scalar("classiifcation/loss_train", loss_train, it*100000+iteration*1000+ep)
            summary_writer.add_scalar("classiifcation/acc_train", acc_train, it*100000+iteration*1000+ep)
        print('Evaluating iteration/epoch: %d training loss: %f training accuracy: %f' % (it*100000+iteration*1000+ep, loss_train, acc_train))
    time_train = time.time() - time_start
    if(logging):
        summary_writer.add_scalar("classiifcation/time_train", time_train, it*100000+iteration*1000+ep)
    loss_val, acc_val = train_classifier(net, device, config, val_loader, "val", optimizer_model, scheduler_model)
    if(logging):
        summary_writer.add_scalar("classiifcation/loss_val", loss_val, it*100000+iteration*1000+ep)
        summary_writer.add_scalar("classiifcation/acc_val", acc_val, it*100000+iteration*1000+ep)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, epoch_eval_train, int(time_train), loss_train, acc_train, acc_val))
    return acc_train, acc_val




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
        logit = net(input)
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
    print(logging)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = configparser.ConfigParser()
    config.read("configs/distillation_config.ini")
    def_conf = config["DEFAULT"]
    ipc = def_conf.getint("n_cad_per_class", 1) ## Number of CAD models per class
    channel = def_conf.getint("channel", 3) ## Should it actually be a channel? 
    num_classes = def_conf.getint("num_classes", 2)
    outer_loop, inner_loop = get_loops(ipc) ## Variable defines the number of models per class
    total_iterations = def_conf.getint("total_iterations")
    eval_iteration_pool = np.arange(0, total_iterations+1, 500).tolist()
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
    indices_class = [[] for c in range(55)]
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

    label_syn = torch.tensor(np.array([np.ones(ipc, dtype=int)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1)


    ### For REAL initialization
    if(def_conf.get("initialization")=="real"):
        print("======= Initializing synthetic dataset from real data ==========")
        for c in range(num_classes):
            path_to_rand_cad = get_rand_cad(c, ipc, indices_class, cad_all_path)
            cad_pts = get_cad_points(path_to_rand_cad, def_conf.getint("num_points"))
            cad_syn.data[c*ipc:(c+1)*ipc] = cad_pts.detach().data
    else:
        print(f"======= Initialized Synthetic dataset with {ipc} CAD per class. CAD array shape is {cad_syn.shape} with values normalized between {cad_syn.min(), cad_syn.max()}===============")

    print("========= Initializing SummaryWriter ==========")
    if(logging):
        summary_writer = SummaryWriter(log_dir=os.path.join(def_conf.get("log_dir"),"distillation"+str(time.time()))) #initialize sumamry writer
    else: summary_writer=None

    ''' training '''
    optimizer_img = torch.optim.SGD([cad_syn, ], lr=def_conf.getfloat("lr_cad", 0.1), momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    # criterion = nn.CrossEntropyLoss().to(device)

    model_eval_pool = ["MINKENGINE"]

    print('%s training begins'%get_time())
    for it in range(total_iterations):
        if it in eval_iteration_pool:
            ## TODO find other possible evaluation models
            for model_eval in model_eval_pool:
                net = MinkowskiFCNN(
                    in_channel=3, out_channel=num_classes, embedding_channel=1024, classification_mode=def_conf.get("classification_mode")
                ).to(device)
                ## TODO: Import weights to the model, and eval and return accuracy
                net.load_state_dict(loaded_dict['state_dict'])
                accs_train = []
                for it_eval in range(def_conf.getint("num_eval")):
                    cad_syn_eval, label_syn_eval = copy.deepcopy(cad_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                    syn_ds = TensorDataset(cad_syn_eval, label_syn_eval)
                    syn_loader = torch.utils.data.DataLoader(syn_ds, batch_size=4, collate_fn=minkowski_collate_fn, drop_last=True)
                    acc_train, acc_test = evaluate_synset(it, it_eval, net, syn_loader, val_loader, def_conf, loaded_dict, device, summary_writer)
                    accs_train.append(acc_train)
                ## TODO maybe decrease the size of the val set.... 

            '''Save point cloud'''
            ## TODO Save point cloud
            ## TODO Make sure that PCD values are still normalized from -1 to 1

    ## TODO Create network for DD
    ## TODO Improved logging. Instead of the difficult calculation. 

        