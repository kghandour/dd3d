from configs import settings
from models.MEConv import MEConv, create_input_batch
from utils.Mnist2D import Mnist2Dreal, Mnist2Dsyn, get_dataset, Mnist2D, get_mnist_dataloader
from torch.utils.data import DataLoader
from utils.MinkowskiCollate import stack_collate_fn, minkowski_collate_fn
import MinkowskiEngine as ME
import os, torch
import numpy as np


def generate_synth(dst_train, num_classes):
    ''' organize the real dataset '''
    global indices_class, images_all, labels_all, image_syn, label_syn

    indices_class = [[] for c in range(num_classes)]
    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to(settings.device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=settings.device)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*settings.cad_per_class, settings.num_points, 3), dtype=torch.float, requires_grad=True, device=settings.device)
    label_syn = torch.tensor(np.array([np.ones(settings.cad_per_class)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

def get_images(c, n): # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    img_real = images_all[idx_shuffle]
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(img_real, labels)
    return get_mnist_dataloader(dataset, n)

def generate_synth_dataloader(c):
    img_syn = image_syn[c*settings.cad_per_class:(c+1)*settings.cad_per_class]
    lab_syn = torch.ones((settings.cad_per_class,), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dsyn(img_syn, lab_syn)
    return get_mnist_dataloader(dataset, settings.cad_per_class)

if __name__ == "__main__":
    settings.init()

    outer_loop, inner_loop = 1, 1
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset()
    train_loader = DataLoader(dst_train, batch_size=settings.batch_size, shuffle=True, collate_fn=minkowski_collate_fn)
    network = MEConv(in_channel=3, out_channel=10).to(settings.device)
    total_iterations = settings.distillationconfig.getint("total_iterations")
    eval_iteration_pool = [total_iterations-1]
    export_cad_dir = os.path.join(settings.distillationconfig.get("export_dir"), settings.exp_file_name)
    os.makedirs(export_cad_dir, exist_ok=True)

    generate_synth(dst_train=dst_train, num_classes=10)

    optimizer_img = torch.optim.SGD([image_syn, ], lr=settings.modelconfig.getfloat("dist_lr"), momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    
    for c in range(num_classes):
        for batch in get_images(c, 8):
            input = create_input_batch(batch, True, device=settings.device)
            loss = network(input)
        for batch in generate_synth_dataloader(c):
            input = create_input_batch(batch, True, device=settings.device)
            loss = network(input)
            # print(loss)
        

