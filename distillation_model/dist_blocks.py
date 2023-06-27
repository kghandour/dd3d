import time
from sklearn import metrics
import torch
from classification_model.me_classification import create_input_batch
from classification_model.me_network import criterion
from classification_model.shapepcd_set import minkowski_collate_fn
from utils.utils import create_loader_for_synthetic_cad, get_time


def run_epoch(network, optimizer, scheduler, phase, dataloader, config, device):
    if phase == "train":
        network.train()
    else:
        network.eval()
    loss_avg, acc_avg, num_exp = 0, 0, 0
    for train_iter in dataloader:
        input = create_input_batch(
            train_iter, device=device, quantization_size=config.getfloat("voxel_size")
        )
        logit = network(input)
        loss = criterion(
            logit, train_iter["labels"].to(device), config.get("classification_mode")
        )
        accuracy = metrics.accuracy_score(
            train_iter["labels"].cpu(), torch.argmax(logit, 1).cpu()
        )

        if phase == "train":
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


def evaluate_classification(
    network,
    optimizer,
    scheduler,
    config,
    cad_syn_tensor,
    label_syn_tensor,
    validation_loader,
    device,
):
    print("====== Testing Classifier =========")
    for it_eval in range(config.getint("num_eval")):
        synthetic_cad_dataloader = create_loader_for_synthetic_cad(
            cad_syn_tensor=cad_syn_tensor,
            label_syn_tensor=label_syn_tensor,
            make_copy=True,
            batch_size=2
        )

        metrics_dict = {
            "loss_train": 0,
            "acc_train": 0,
            "loss_val": 0,
            "acc_val": 0,
            "time_train": 0,
        }
        epoch_eval_train = config.getint("epoch_eval_train")
        time_start = time.time()

        ## Run the synthetic cad through the pretrained classification network for a train run (Modifies weights)
        for ep in range(epoch_eval_train + 1):
            metrics_dict["loss_train"], metrics_dict["acc_train"] = run_epoch(
                network,
                optimizer,
                scheduler,
                phase="train",
                dataloader=synthetic_cad_dataloader,
                config=config,
                device=device,
            )

        metrics_dict["time_train"] = time.time() - time_start

        ## Runs the real validation set through the pretrained classification network for validation (Doesn't modify weights)
        metrics_dict["loss_val"], metrics_dict["acc_val"] = run_epoch(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            phase="val",
            dataloader=validation_loader,
            config=config,
            device=device,
        )

        print(
            "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
            % (
                get_time(),
                it_eval,
                epoch_eval_train,
                int(metrics_dict["time_train"]),
                metrics_dict["loss_train"],
                metrics_dict["acc_train"],
                metrics_dict["acc_val"],
            )
        )

        return metrics_dict