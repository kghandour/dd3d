## Starting fresh
## File created by Karim

import os, sys, argparse
import configs.settings as settings
import torch
from utils.ShapeNet import ShapeNetDataset
from torch.utils.data import DataLoader
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model, get_loss, get_ce_loss
import numpy as np
from tqdm import tqdm
from test_classification import test_classification
from utils.SyntheticDataset import init_synth, SyntheticDataset
from distillation_loss import get_distillation_loss, match_loss
import torch.nn as nn
from utils.train_val_split import CLASS_NAME_TO_ID
import math
import shutil
from utils.Network import get_network


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

if __name__=="__main__":
    settings.init()

    cls_list = ["airplane"]
    outer_loop, inner_loop = 1, 1
    ShapenetDS = ShapeNetDataset(modelnet40=False, phase="val", cls_list=cls_list)
    ShapenetDataloader = DataLoader(ShapenetDS, batch_size=settings.modelconfig.getint("batch_size"), num_workers=settings.num_workers, drop_last=True)
    classifier_network = get_model(settings.num_classes, normal_channel=False).to(device=settings.device)
    checkpoint = torch.load(os.path.join(settings.modelconfig.get("checkpoint_dir"),'20230717230958_ssg32_sgd.pth'))
    classifier_network.load_state_dict(checkpoint['model_state_dict'])
    export_cad_dir = os.path.join(settings.distillationconfig.get("export_dir"), settings.exp_file_name)
    os.makedirs(export_cad_dir, exist_ok=True)
    total_iterations = settings.distillationconfig.getint("total_iterations")
    eval_iteration_pool = [total_iterations-1]

    synthetic_xyz, synthetic_labels = init_synth(cls_list=cls_list)

    synthetic_optimizer = settings.get_optimizer([synthetic_xyz], target="dist", opt="adam")
    distillation_network = get_network("convnet").to(settings.device)
    load_dist_path = settings.distillationconfig.get("load_dist_model", "None")
    if(load_dist_path != "None"):
        settings.log_string("Loading pretrained Distilled state from "+load_dist_path)
        load_dist_dict = torch.load(load_dist_path)
        distillation_network.load_state_dict(load_dist_dict["model_state_dict"])

    distillation_criterion = get_ce_loss()
    for m in distillation_network.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
        if isinstance(m, nn.Dropout):
            m.eval()
    loop_n = len(cls_list) if len(cls_list) > 0 else settings.num_classes
    log_loss = []

    least_loss = math.inf
    if(len(cls_list)==1):
        temp_path = settings.get_fixed_cad_path( CLASS_NAME_TO_ID[cls_list[0]])
        settings.log_string("Exporting the main model for comparison: "+str(temp_path[0]))
        shutil.copy(temp_path[0], "/home/ghandour/")

    for iteration in tqdm(range(total_iterations),desc="Iteration: ", smoothing=0.9):
        
        if(iteration in eval_iteration_pool):
            with torch.no_grad():
                synthetic_dataset = SyntheticDataset(synthetic_xyz, synthetic_labels)
                SyntheticDataLoader = DataLoader(
                    dataset=synthetic_dataset,
                    batch_size=settings.batch_size,
                    shuffle=False
                )
                synthetic_instance_acc, synthetic_class_acc = test_classification(classifier_network.eval(), SyntheticDataLoader, vote_num=settings.modelconfig.getint("num_votes"), num_class=settings.num_classes)
                instance_acc, class_acc = test_classification(classifier_network.eval(), ShapenetDataloader, vote_num=settings.modelconfig.getint("num_votes"), num_class=settings.num_classes)
                settings.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                settings.log_string('Synthetic Instance Accuracy: %f, Synthetic Class Accuracy: %f' % (synthetic_instance_acc, synthetic_class_acc))

        for ol in range(outer_loop):
            for cls_iter in range(loop_n):
                if(len(cls_list)>0):
                    cls = CLASS_NAME_TO_ID[cls_list[cls_iter]]
                else:
                    cls = cls_iter
                synthetic_optimizer.zero_grad()
                RealDataLoader = settings.get_fixed_cad_loader(cls)
                synthetic_cls_xyz = synthetic_xyz[cls_iter * settings.cad_per_class : (cls_iter + 1) * settings.cad_per_class].clone()
                synthetic_cls_label = torch.ones((settings.cad_per_class), device=settings.device, dtype=torch.long) * cls
                synthetic_cls_dataset = SyntheticDataset(synthetic_cls_xyz, synthetic_cls_label)
                SyntheticClassDataloader = DataLoader(
                    dataset=synthetic_cls_dataset,
                    batch_size=settings.batch_size,
                    shuffle=False
                )
                
                RealLoss = get_distillation_loss(distillation_network=distillation_network, distillation_criterion=distillation_criterion, dataloader=RealDataLoader)
                gw_real = torch.autograd.grad(RealLoss, distillation_network.parameters())
                gw_real = list((_.detach().clone() for _ in gw_real))
                SyntheticLoss = get_distillation_loss(distillation_network, distillation_criterion,SyntheticClassDataloader)
                gw_syn = torch.autograd.grad(
                    SyntheticLoss, distillation_network.parameters(), create_graph=True
                )
                loss = match_loss(gw_syn, gw_real, dis_metric="ours", device=settings.device)

                log_loss.append(loss.item())
                loss.backward()
                synthetic_optimizer.step()
            if ol == outer_loop - 1:
                break
        if(iteration%10 ==0):
            loss_value = sum(log_loss)/len(log_loss)
            settings.log_string("Iteration: "+str(iteration)+" Loss matched is "+ str(loss_value))
            settings.log_tensorboard("Distillation/Matched Loss",loss_value, iteration)
            log_loss = []
        if(iteration%settings.distillationconfig.getint("save_cad_every")==0):
            if (loss_value <= least_loss):
                settings.log_string("Saving Model")
                settings.log_string("New Loss:"+str(loss_value)+" Old Loss: "+str(least_loss))
                least_loss = loss_value
                savepath = os.path.join(settings.distillationconfig.get("distillation_checkpoint_dir"), settings.exp_file_name+".pth")
                settings.log_string('Saving at %s' % savepath)
                state = {
                    'epoch': iteration+1,
                    'least_loss': least_loss,
                    'model_state_dict': distillation_network.state_dict(),
                    'optimizer_state_dict': synthetic_optimizer.state_dict(),
                    'synthetic_xyz': synthetic_xyz.clone(),
                    'synthetic_labels': synthetic_labels.clone()
                }
                torch.save(state, savepath)

            settings.log_string("Exporting Point Cloud")
            settings.save_cad(synthetic_xyz, export_cad_dir, iteration)

