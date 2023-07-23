from configs import settings
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model, get_loss, get_ce_loss
from test_classification import test_classification
from utils.ShapeNet import ShapeNetDataset
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import models.pointnet2_ssg_wo_normals.provider as provider
import numpy as np

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train_classifier(classifier_network, classifier_optimizer, classifier_criterion):
    max_epochs = settings.modelconfig.getint("epoch", 200)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    for epoch in range(start_epoch, max_epochs):
        settings.log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, max_epochs))
        mean_correct = []
        loss_list = []
        classifier_network = classifier_network.train()
        for batch_id, (points, target) in tqdm(enumerate(train_ShapenetDataloader, 0), total=len(train_ShapenetDataloader), smoothing=0.9):
            classifier_optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(settings.device), target.to(settings.device)

            pred, trans_feat = classifier_network(points)
            loss = classifier_criterion(pred, target.long(), trans_feat)
            if(global_step%100==0): settings.log_string("Train loss at global step "+str(global_step)+":"+str(loss.item()))
            settings.log_tensorboard("Classifier/batch_loss",loss.item(),global_step)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss_list.append(loss.item())
            loss.backward()
            classifier_optimizer.step()
            global_step += 1

        loss_mean = sum(loss_list)/len(loss_list)
        train_instance_acc = np.mean(mean_correct)
        settings.log_string("Train Loss"+ str(loss_mean))
        settings.log_tensorboard("Classifier/epoch_loss",loss_mean,global_step)
        settings.log_tensorboard("Classifier/train_instance_acc", train_instance_acc, global_step)
        settings.log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test_classification(classifier_network.eval(), test_ShapenetDataloader, vote_num=settings.modelconfig.getint("num_votes"), num_class=settings.num_classes)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            settings.log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            settings.log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            settings.log_tensorboard("Classifier/Test_Instance_Acc", instance_acc, global_step)
            settings.log_tensorboard("Classifier/Test_Class_Acc", class_acc, global_step)
            settings.log_tensorboard("Classifier/Test_Best_Instance_Acc", best_instance_acc, global_step)
            settings.log_tensorboard("Classifier/Test_Best_Class_Acc", best_class_acc, global_step)


            if (instance_acc >= best_instance_acc):
                settings.logger.info('Save model...')
                savepath = ckpt_path
                settings.log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier_network.state_dict(),
                    'optimizer_state_dict': classifier_optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    settings.logger.info('End of training...')
    


if __name__=="__main__":
    settings.init()
    os.makedirs(settings.modelconfig.get("checkpoint_dir"), exist_ok=True)
    ckpt_path = os.path.join(settings.modelconfig.get("checkpoint_dir"),settings.exp_file_name+".pth")
    if(settings.modelconfig.get("load_ckpt_name", "None")!="None"):
        ckpt_path = os.path.join(settings.modelconfig.get("checkpoint_dir"),settings.modelconfig.get("load_ckpt_name", "None"))
    cls_list = []

    '''Initializing Datasets and loaders'''
    train_ShapenetDS = ShapeNetDataset(modelnet40=False, phase="train", cls_list=cls_list)
    train_ShapenetDataloader = DataLoader(train_ShapenetDS, batch_size=settings.modelconfig.getint("batch_size"), num_workers=settings.num_workers, drop_last=True, shuffle=True)
    
    test_ShapenetDS = ShapeNetDataset(modelnet40=False, phase="val", cls_list=cls_list)
    test_ShapenetDataloader = DataLoader(test_ShapenetDS, batch_size=settings.modelconfig.getint("batch_size"), num_workers=settings.num_workers, drop_last=True)
    

    '''Initializing Model'''
    classifier_network = get_model(settings.num_classes, normal_channel=False).to(device=settings.device)
    classifier_criterion = get_ce_loss().to(device=settings.device)
    classifier_network.apply(inplace_relu)

    try:
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        classifier_network.load_state_dict(checkpoint['model_state_dict'])
        settings.log_string('Use pretrain model')
    except:
        settings.log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    
    classifier_optimizer = settings.get_optimizer(classifier_network.parameters(), target="classifier", opt=settings.modelconfig.get("dist_opt"))

    '''Starting Training'''
    settings.log_string("Starting training")
    train_classifier(classifier_network, classifier_optimizer, classifier_criterion)