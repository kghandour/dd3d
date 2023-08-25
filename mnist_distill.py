from configs import settings
from distillation_loss import match_loss
from models.MEConv import MEConv, create_input_batch, MEConvImage
from utils.Mnist2D import Mnist2Dreal, Mnist2Dsyn, get_dataset, Mnist2D, get_mnist_dataloader
from torch.utils.data import DataLoader
from utils.MinkowskiCollate import stack_collate_fn, minkowski_collate_fn
import MinkowskiEngine as ME
import os, torch
import numpy as np
import torch.nn as nn
import math
import open3d as o3d
from torchinfo import summary as torchsummary
import copy
from torchvision.utils import save_image
import torch.nn.functional as F
import time
# def generate_synth(dst_train, num_classes):
#     ''' organize the real dataset '''
#     global indices_class, images_all, labels_all, image_syn, label_syn

#     indices_class = [[] for c in range(num_classes)]
#     images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
#     labels_all = [dst_train[i][1] for i in range(len(dst_train))]
#     for i, lab in enumerate(labels_all):
#         indices_class[lab].append(i)
#     images_all = torch.cat(images_all, dim=0).to(settings.device)
#     labels_all = torch.tensor(labels_all, dtype=torch.long, device=settings.device)

#     for c in range(num_classes):
#         print('class c = %d: %d real images'%(c, len(indices_class[c])))

#     for ch in range(channel):
#         print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

#     ''' initialize the synthetic data '''
#     image_syn = torch.randn(size=(num_classes*settings.cad_per_class, settings.num_points, 3), dtype=torch.float, requires_grad=True, device=settings.device)
#     label_syn = torch.tensor(np.array([np.ones(settings.cad_per_class)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

def export_pcd(arr, c, custom_name="orig"):
    to_save = np.asarray(arr.cpu()[0])
    # print(to_save)
    # occ_grid = np.argwhere(to_save==1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_save)
    name = str(c)
    o3d.io.write_point_cloud(
        export_cad_dir
        + "/original-"
        + custom_name
        +"-"
        + name
        + "_"
        + str(iteration)
        + ".ply",
        pcd,
    )

def get_images(c, n): # get random n images from class c
    # print(c)
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    img_real = images_all[idx_shuffle]
    # print("Indices for input digit %i, %i",c, idx_shuffle)
    # print(img_real.shape)
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(img_real, labels, pixel_val)
    return get_mnist_dataloader(dataset, n)

def get_images_fixed(c,idx, n): # get random n images from class c
    # print(c)
    # print(idx)
    # idx_shuffle = np.random.permutation(indices_class[c])[:n]
    img_real = images_all[indices_class[c][idx]]
    print(img_real.shape)
    # print("Indices for input digit %i, %i",c, idx_shuffle)
    # print(img_real.shape)
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(torch.unsqueeze(img_real, 0), labels, pixel_val)
    return get_mnist_dataloader(dataset, n)

def get_1_pt_fixed(c, n):
    img_real = torch.tensor([[[0, 14, 14]]], device=settings.device)
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(img_real, labels, pixel_val, fixed_pt=True)
    # export_pcd(img_real, c, custom_name="1pt")
    return get_mnist_dataloader(dataset, n)

def get_syn_tensor(c):
    img_syn = image_syn[c*settings.cad_per_class:(c+1)*settings.cad_per_class]
    lab_syn = torch.ones((settings.cad_per_class,), device=settings.device, dtype=torch.long) * c
    return img_syn, lab_syn

def generate_synth_dataloader(c):
    img_syn, lab_syn = get_syn_tensor(c)
    dataset = Mnist2Dsyn(img_syn, lab_syn, pixel_val)
    return get_mnist_dataloader(dataset, settings.cad_per_class)

if __name__ == "__main__":
    settings.init()
    global export_cad_dir
    global pixel_val

    pixel_val = False 
    torch.random.manual_seed(int(time.time() * 1000) % 100000)

    outer_loop, inner_loop = 1, 1
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(pixel_val)
    num_classes = 1
    train_loader = DataLoader(dst_train, batch_size=settings.batch_size, shuffle=True, collate_fn=minkowski_collate_fn)
    if(pixel_val): in_c = 1 
    else: in_c = 3
    network = MEConvImage(in_channel=in_c, out_channel=10).to(settings.device)
    torchsummary(network)
    settings.log_string(network)
    total_iterations = settings.distillationconfig.getint("total_iterations")
    eval_iteration_pool = [total_iterations-1]
    export_cad_dir = os.path.join(settings.distillationconfig.get("export_dir"), settings.exp_file_name)
    os.makedirs(export_cad_dir, exist_ok=True)

    indices_class = [[] for c in range(10)] ## TODO: CHANGE when SHAPENET
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
    syn_shape = (num_classes*settings.cad_per_class, settings.num_points, 3)
    if(pixel_val):
        image_syn = torch.randn(size=(num_classes*settings.cad_per_class, 1, im_size[0], im_size[1]), dtype=torch.float, device=settings.device, requires_grad=True)
        # image_syn = F.normalize(image_syn).requires_grad_()
        # image_syn = torch.randn(size=(num_classes*settings.cad_per_class, im_size[0] * im_size[1], 1), dtype=torch.float, requires_grad=True, device=settings.device)
    else:
        image_syn = (28 - 0) * torch.rand(size= syn_shape, dtype=torch.float, device=settings.device)
        # image_syn = torch.dstack((torch.zeros((image_syn.shape[0], image_syn.shape[1], 1)).to(settings.device), image_syn)).requires_grad_()

    # while(image_syn.unique(dim=1).shape != syn_shape):
    #     image_syn = torch.randint(0, 28, size= syn_shape, dtype=torch.float, device=settings.device)
    #     settings.log_string("Repeated elements found, regenerating again")
    label_syn = torch.tensor(np.array([np.ones(settings.cad_per_class)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    criterion = nn.CrossEntropyLoss().to(settings.device)
    optimizer_img = torch.optim.SGD([image_syn, ], lr=settings.modelconfig.getfloat("dist_lr"), momentum=0.9) # optimizer_img for synthetic data
    # optimizer_img = torch.optim.Adam([image_syn, ], lr=settings.modelconfig.getfloat("dist_lr"), betas=(0.9, 0.999)) # optimizer_img for synthetic data
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_img, T_max=settings.distillationconfig.getint("total_iterations"))
    optimizer_img.zero_grad()
    net_parameters = list(network.parameters())
    log_loss = []
    least_loss = math.inf
    loss_avg = 0
    for iteration in range(settings.distillationconfig.getint("total_iterations")):
        loss_avg = 0
        if(not pixel_val):
            image_syn.requires_grad_(False)
            image_syn[image_syn<0]=0
            image_syn[image_syn>27]=27
            image_syn[:,:,0] = 0
            image_syn.requires_grad_()
        for ol in range(outer_loop):
            loss = torch.tensor(0.0).to(settings.device)
            for c in range(num_classes):
                # for batch in get_images(c, settings.modelconfig.getint("batch_size")):
                # for batch in get_images_fixed(c, 0, settings.modelconfig.getint("batch_size")):
                for batch in get_1_pt_fixed(c, settings.modelconfig.getint("batch_size")):
                    input = create_input_batch(batch, True, device=settings.device, quantization_size=1)
                    # print(input.shape)
                    # print(np.max(input.coordinates.clone().cpu().numpy(), keepdims=True))
                    output = network(input)
                    # print(output.shape)
                    loss_real = criterion(output, batch['labels'])
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                for batch in generate_synth_dataloader(c):
                    input = create_input_batch(batch, True, device=settings.device, quantization_size=1)
                    output = network(input)
                    loss_syn = criterion(output, batch['labels'])
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                loss += match_loss(gw_syn, gw_real, settings.modelconfig.get("dist_opt"), settings.device)
            optimizer_img.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_([image_syn, ], 1.0)
            optimizer_img.step()
            # scheduler.step()
            loss_avg += loss.item()
            # torch.cuda.empty_cache()

        loss_avg /= (num_classes*outer_loop)
        log_loss.append(loss_avg)
        # settings.log_string("Iteration "+str(iteration)+" Loss:"+str(loss_avg))
        # settings.log_tensorboard("Distillation/Matched Loss",loss_avg, iteration)
        if(iteration%100 ==0):
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
                # settings.log_tensorboard_pcd('Point Clouds', get_syn_tensor(0), iteration)
                state = {
                    'epoch': iteration+1,
                    'least_loss': least_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer_img.state_dict(),
                    'synthetic_xyz': image_syn.clone(),
                    'synthetic_labels': label_syn.clone()
                }
                torch.save(state, savepath)

            settings.log_string("Exporting Point Cloud")
            if(not pixel_val):
                settings.save_cad(image_syn, export_cad_dir, iteration, normalize=False)
            else:
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                # reshaped = image_syn.reshape(num_classes, settings.cad_per_class, 28, 28)
                save_name = os.path.join(export_cad_dir, 'vis_iter%d.png'%(iteration))
                save_image(image_syn_vis, save_name, nrow=settings.cad_per_class)

