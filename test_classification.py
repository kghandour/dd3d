import numpy as np
from tqdm import tqdm
import torch
def test_classification(model, loader, num_class=40, vote_num=1):
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
    class_acc = np.mean(class_acc[:, 2] ,where=class_acc[:, 2]!=0)
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc