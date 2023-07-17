## Starting fresh
## File created by Karim

import os, sys, argparse
import configs.settings as settings
import torch
from utils.ShapeNet import ShapeNetDataset
from torch.utils.data import DataLoader
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model, get_loss
import numpy as np
from tqdm import tqdm
from test_classification import test_classification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

if __name__=="__main__":
    settings.init()
    
    ShapenetDS = ShapeNetDataset(modelnet40=True, phase="val", cls_list=[])
    ShapenetDataloader = DataLoader(ShapenetDS, batch_size=settings.modelconfig.getint("batch_size"), num_workers=settings.num_workers, drop_last=True)
    classifier_network = get_model(settings.num_classes, normal_channel=False).to(device=settings.device)
    checkpoint = torch.load(os.path.join("checkpoint_dir",'best_model.pth'))
    classifier_network.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test_classification(classifier_network.eval(), ShapenetDataloader, vote_num=settings.modelconfig.getint("num_votes"), num_class=settings.num_classes)
        settings.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
