import torch
from classification_model.me_classification import create_input_batch
from utils.utils import RealTensorDataset, create_loader_for_synthetic_cad, get_rand_cad, match_loss
from classification_model.shapepcd_set import class_id, minkowski_collate_fn
import torch.nn.functional as F


def get_distillation_loss(
    input_real_loader,
    device,
    def_conf,
    distillation_network,
    net_parameters,
    cad_syn_tensor,
    label_syn_tensor,
    batch_size=4
):
    loss_real_list = []
    loss_syn_list = []

    for input_real_iter in input_real_loader:
        input_real = create_input_batch(
            input_real_iter,
            device=device,
            quantization_size=def_conf.getfloat("voxel_size"),
        )
        output_real = distillation_network(input_real)
        loss_real = F.cross_entropy(
            output_real,
            input_real_iter["labels"].to(device),
            reduction="mean",
        )
        loss_real_list.append(loss_real)
    loss_real = sum(loss_real_list) / len(loss_real_list)
    gw_real = torch.autograd.grad(loss_real, net_parameters)
    gw_real = list((_.detach().clone() for _ in gw_real))

    syn_loader = create_loader_for_synthetic_cad(cad_syn_tensor, label_syn_tensor, make_copy=False, batch_size=batch_size)
    for input_syn_iter in syn_loader:
        input_syn = create_input_batch(
            input_syn_iter,
            device=device,
            quantization_size=def_conf.getfloat("voxel_size"),
        )
        output_syn = distillation_network(input_syn)
        loss_syn = F.cross_entropy(
            output_syn,
            input_syn_iter["labels"].to(device),
            reduction="mean",
        )
        loss_syn_list.append(loss_syn)
    loss_syn = sum(loss_syn_list) / len(loss_syn_list)
    gw_syn = torch.autograd.grad(
        loss_syn, net_parameters, create_graph=True
    )

    return match_loss(
        gw_syn, gw_real, def_conf.get("loss_method"), device=device
    )


def outer_block(
    num_classes,
    indices_class,
    cad_all_path,
    device,
    cad_syn_tensor,
    label_syn_tensor,
    ipc,
    def_conf,
    distillation_network,
    net_parameters,
    optimizer_distillation,
    batch_size=4,
    classes_to_distill=[],
):
    loss = torch.tensor(0.0).to(device)
    if len(classes_to_distill) > 0:
        for idx, c in enumerate(classes_to_distill):
            cad_real_class = get_rand_cad(class_id[c], 4, indices_class, cad_all_path)
            labels_real_class = (
                torch.ones((cad_real_class.shape[0],), device=device, dtype=torch.long)
                * class_id[c]
            )
            cad_syn_class = cad_syn_tensor[idx * ipc : (idx + 1) * ipc].clone()
            label_syn_class = (
                torch.ones((ipc), device=device, dtype=torch.long) * class_id[c]
            )
            input_real_ds = RealTensorDataset(cad_real_class, labels_real_class)
            input_real_loader = torch.utils.data.DataLoader(
                input_real_ds,
                batch_size=batch_size,
                collate_fn=minkowski_collate_fn,
                drop_last=True,
            )

            loss += get_distillation_loss(
                input_real_loader=input_real_loader,
                device=device,
                def_conf=def_conf,
                distillation_network=distillation_network,
                net_parameters=net_parameters,
                cad_syn_tensor=cad_syn_class,
                label_syn_tensor=label_syn_class,
                batch_size=batch_size
            )
    else:
        for c in range(num_classes):
            cad_real_class = get_rand_cad(c, 4, indices_class, cad_all_path)
            labels_real_class = (
                torch.ones(
                    (cad_real_class.shape[0],), device=device, dtype=torch.long
                )
                * c
            )
            cad_syn_class = cad_syn_tensor[c * ipc : (c + 1) * ipc].clone()
            label_syn_class = torch.ones((ipc), device=device, dtype=torch.long) * c
            input_real_ds = RealTensorDataset(cad_real_class, labels_real_class)
            input_real_loader = torch.utils.data.DataLoader(
                input_real_ds,
                batch_size=batch_size,
                collate_fn=minkowski_collate_fn,
                drop_last=True,
            )

            loss += get_distillation_loss(
                input_real_loader=input_real_loader,
                device=device,
                def_conf=def_conf,
                distillation_network=distillation_network,
                net_parameters=net_parameters,
                cad_syn_tensor=cad_syn_class,
                label_syn_tensor=label_syn_class,
            )
    optimizer_distillation.zero_grad()
    loss.backward()
    optimizer_distillation.step()
    return loss.item()
