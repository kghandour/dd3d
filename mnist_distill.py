from configs import settings
from distillation_loss import match_loss
from models.MEConv import MEConv, create_input_batch, MEConvImage, MEConvExp, MEPytorch
from utils.Mnist2D import Mnist2Dreal, Mnist2Dsyn, get_dataset, Mnist2D, get_mnist_dataloader, TensorDataset
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
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import time
from models.MNIST import ConvNet
from scipy.ndimage.interpolation import rotate as scipyrotate


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

def export_mnist(arr, c, custom_name="orig"):
    cad_list_copy = arr.clone().detach().cpu().numpy()
    cad_list_copy = np.array([cad_list_copy])
    for i, cad in enumerate(cad_list_copy):
        cad = np.argwhere(cad==1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cad)
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

def augment(images, device):
    # This can be sped up in the future.
    dc_aug_param = {'crop': 4, 'scale': 0.2, 'rotate': 45, 'noise': 0.001, 'strategy': 'crop_scale_rotate'}
    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images

def epoch(mode, dataloader, net, optimizer, criterion, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(settings.device)
    criterion = criterion.to(settings.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(settings.device)
        img = augment(img, settings.device)
        lab = datum[1].long().to(settings.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def populate_img(occ_grid, img_array):
    for idx, digit in enumerate(occ_grid):
        for pt in digit:
            newpt = pt.type(torch.LongTensor)
            img_array[idx,0,newpt[1],newpt[2]]=1
    return img_array
def evaluate_synset(it_eval, net, images_train, labels_train, testloader, iteration):
    net = net.to(settings.device)
    images_train = images_train.to(settings.device)
    labels_train = labels_train.to(settings.device)
    lr = float(settings.modelconfig.getfloat("classifier_lr"))
    Epoch = 1000
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(settings.device)
    mnist_imgs = torch.zeros(size=(images_train.shape[0], 1, 28, 28), dtype=torch.float, device=settings.device)
    mnist_imgs = populate_img(images_train, mnist_imgs)
    save_name = os.path.join(export_cad_dir, 'vis_iter%d.png'%(iteration))
    save_image(mnist_imgs, save_name, nrow=settings.cad_per_class)
    if(it_eval==0):
        settings.log_tensorboard_image("Distillation/Images", make_grid(mnist_imgs, nrow=settings.cad_per_class), iteration)
    dst_train = TensorDataset(mnist_imgs, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, aug = False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, aug = False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test

def get_images(c, n): # get random n images from class c
    # print(c)
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    img_real = images_all[idx_shuffle]
    # print("Indices for input digit %i, %i",c, idx_shuffle)
    # print(img_real.shape)
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(img_real, labels, pixel_val)
    return get_mnist_dataloader(dataset, n)

def get_images_fixed(c,idx, n, export_cad_once): # get random n images from class c
    # print(c)
    # print(idx)
    # idx_shuffle = np.random.permutation(indices_class[c])[:n]
    img_real = images_all[indices_class[c][idx]]
    if(export_cad_once):
        export_mnist(img_real, c, custom_name="2pt")
    # print("Indices for input digit %i, %i",c, idx_shuffle)
    # print(img_real.shape)
    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(torch.unsqueeze(img_real, 0), labels, pixel_val)
    return get_mnist_dataloader(dataset, n)

def get_1_pt_fixed(c, n, export_cad_once):
    # img_real = torch.tensor([[[0, 1, 1]]], device=settings.device)
    # img_real = torch.tensor([[[0, 3, 3], [0, 6, 6], [0, 3, 6]]], device=settings.device) # Working config 
    # img_real = torch.tensor([[[0, 10, 10], [0, 20, 20], [0, 10, 20]]], device=settings.device) # Not working
    # img_real = torch.tensor([[[0, 10, 10], [0, 20, 20], [0, 10, 20]]], device=settings.device) # Not working
    # img_real = torch.tensor([[[0, 0.5, 0.5]]], device=settings.device) #  working
    # img_real = torch.tensor([[[0, 20.0, 20.0]]], device=settings.device) # Working with S: 0.1 
    img_real = torch.tensor([[[0, 0.7, 0.7], [0, 0.3, 0.3]]], device=settings.device)


    # img_real = torch.tensor([[[0, 3, 3]]], device=settings.device)

    labels = torch.ones((img_real.shape[0],), device=settings.device, dtype=torch.long) * c
    dataset = Mnist2Dreal(img_real, labels, pixel_val, fixed_pt=True)
    if(export_cad_once):
        export_pcd(img_real, c, custom_name="2pt")
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
    export_cad_once = True
    minkpyt = False
    settings.Pyt = minkpyt
    normalized = False
    torch.random.manual_seed(int(time.time() * 1000) % 100000)

    outer_loop, inner_loop = 1, 1
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(pixel_val)
    num_classes = 10
    train_loader = DataLoader(dst_train, batch_size=settings.batch_size, shuffle=True, collate_fn=minkowski_collate_fn)
    if(pixel_val): in_c = 1 
    else: in_c = 3
    if(minkpyt):
        network = MEPytorch(in_c, 10).to(settings.device)
    else:
        network = MEConvExp(in_channel=in_c, out_channel=10).to(settings.device)
    torchsummary(network)
    network.train()
    settings.log_string(network)
    total_iterations = settings.distillationconfig.getint("total_iterations")
    eval_iteration_pool = [0, 250, total_iterations//4, total_iterations//2, total_iterations//4+total_iterations//2, total_iterations-1]
    export_cad_dir = os.path.join(settings.logging_folder_name, settings.distillationconfig.get("export_folder_name"))
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
        # image_syn = torch.tensor([[[0,0,0], [0,10,10]]], dtype=torch.float, device=settings.device)
        # image_syn = torch.tensor([[[0,1,1], [0, 3, 9], [0,9,9]]], dtype=torch.float, device=settings.device) # Working configuration
        # image_syn = torch.tensor([[[0,1,1], [0, 27, 27], [0,15,15]]], dtype=torch.float, device=settings.device) # NOT Working
        # image_syn = torch.tensor([[[0,1,1], [0, 2, 2], [0,1,2]]], dtype=torch.float, device=settings.device) # NOT Working
        # image_syn = torch.tensor([[[0,0.1,0.1]]], dtype=torch.float, device=settings.device) # Working
        # image_syn = torch.tensor([[[0,0.1,0.1], [0, 0.9, 0.9]]], dtype=torch.float, device=settings.device) # 



        # image_syn = torch.rand(size= syn_shape, dtype=torch.float, device=settings.device)
        # image_syn = torch.dstack((torch.zeros((image_syn.shape[0], image_syn.shape[1], 1)).to(settings.device), image_syn)).requires_grad_()

    # while(image_syn.unique(dim=1).shape != syn_shape):
    #     image_syn = torch.randint(0, 28, size= syn_shape, dtype=torch.float, device=settings.device)
    #     settings.log_string("Repeated elements found, regenerating again")
    label_syn = torch.tensor(np.array([np.ones(settings.cad_per_class)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=settings.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    criterion = nn.CrossEntropyLoss().to(settings.device)
    optimizer_img = torch.optim.SGD([image_syn, ], lr=settings.modelconfig.getfloat("dist_lr"), momentum=0.5) # optimizer_img for synthetic data
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
            if(not normalized):
                image_syn[image_syn<0]=0
                image_syn[image_syn>27]=27
            image_syn[:,:,0] = 0
            image_syn.requires_grad_()

        ## Implement the Evaluation classifier here

        if(iteration in eval_iteration_pool):
        # if(iteration in [0]):
            settings.log_string("----- Evaluation -----")
            settings.log_string("Model_train = Convnet2D")
            settings.log_string("Iteration: "+str(iteration))
            accs = []
            for it_eval in range(5):
                classifier_network = ConvNet(1, 10, 128, 3, 'relu', 'instancenorm', 'avgpooling')
                image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                _, acc_train, acc_test = evaluate_synset(it_eval, classifier_network, image_syn_eval, label_syn_eval, testloader, iteration)
                accs.append(acc_test)
                settings.log_tensorboard('Classifier/Acc Test Eval '+str(it_eval), acc_test, iteration)
            

        for ol in range(outer_loop):
            loss = torch.tensor(0.0).to(settings.device)
            for c in range(num_classes):
                for batch in get_images(c, settings.modelconfig.getint("batch_size")):
                # for batch in get_images_fixed(c, 0, settings.modelconfig.getint("batch_size"), export_cad_once):
                # for batch in get_1_pt_fixed(c, settings.modelconfig.getint("batch_size"), export_cad_once):
                    input = create_input_batch(batch, True, device=settings.device, quantization_size=1)
                    if(iteration%100 == 0 and normalized): settings.log_string(input)
                    # print(input.shape)
                    # print(np.max(input.coordinates.clone().cpu().numpy(), keepdims=True))
                    output = network(input)
                    # print(output.shape)
                    loss_real = criterion(output, batch['labels'])
                    if(iteration %100 == 0): 
                        settings.log_tensorboard_str("Distillation/Real Digit:"+str(c)+" Output", str(output), iteration)
                        settings.log_tensorboard("Distillation/Real Digit:"+str(c)+" Loss", loss_real.item(), iteration)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    # print(input)
                    # print(output)
                    # print(loss_real)
                for batch in generate_synth_dataloader(c):
                    input = create_input_batch(batch, True, device=settings.device, quantization_size=1)
                    if(iteration%100 == 0 and normalized): settings.log_string(input)
                    output = network(input)
                    loss_syn = criterion(output, batch['labels'])
                    if(iteration %100 == 0): 
                        settings.log_tensorboard_str("Distillation/Synthetic Digit:"+str(c)+" Output", str(output), iteration)
                        settings.log_tensorboard("Distillation/Synthetic Digit:"+str(c)+" Loss", loss_syn.item(), iteration)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    # print(input)
                    # print(output)
                    # print(loss_syn)
                matched_loss = match_loss(gw_syn, gw_real, settings.modelconfig.get("dist_opt"), settings.device)
                loss += matched_loss
                settings.log_tensorboard("Matched Loss/ Digit "+ str(c), matched_loss.item(), iteration)
            export_cad_once = False
            optimizer_img.zero_grad()
            loss.backward()
            if(iteration%100==0): settings.log_tensorboard_str('Image Syn grad:', str(image_syn.grad), iteration)
            if(iteration%100==0): settings.log_tensorboard("Distillation/Mean Gradient", torch.mean(torch.abs(image_syn.grad)), iteration)
            old_image_syn = image_syn.clone().detach()
            optimizer_img.step()
            if(iteration%100==0): 
                settings.log_tensorboard_str('Image Syn new point:', str(image_syn), iteration)
                settings.log_tensorboard("Distillation/Mean Point Absolute difference", torch.mean(torch.abs(image_syn - old_image_syn)), iteration)
            # print(image_syn)
            # exit()
            # scheduler.step()
            loss_avg += loss.item()
            torch.cuda.empty_cache()

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

