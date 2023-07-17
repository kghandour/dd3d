## Starting fresh
## File created by Karim

import os, sys, argparse
import configs.settings as settings
import torch
from utils.ShapeNet import ShapeNetDataset
from torch.utils.data import DataLoader
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = np.divide(class_acc[:, 0],class_acc[:, 1],out=np.zeros_like(class_acc[:, 0]), where=class_acc[:, 1]!=0)
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

if __name__=="__main__":
    settings.init()
    
    ShapenetDS = ShapeNetDataset(modelnet40=True, phase="val")
    ShapenetDataloader = DataLoader(ShapenetDS, batch_size=settings.modelconfig.getint("batch_size"), num_workers=settings.num_workers, drop_last=True)
    classifier_network = get_model(settings.num_classes, normal_channel=False).to(device=settings.device)
    checkpoint = torch.load('/home/ghandour/dd3d/models/pointnet2_ssg_wo_normals/checkpoints/best_model.pth')
    classifier_network.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier_network.eval(), ShapenetDataloader, vote_num=settings.modelconfig.getint("num_votes"), num_class=settings.num_classes)
        settings.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
