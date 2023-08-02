from configs import settings
from models.MEConv import MEConv, create_input_batch
from utils.Mnist2D import get_dataset, Mnist2D
from torch.utils.data import DataLoader
from utils.MinkowskiCollate import stack_collate_fn, minkowski_collate_fn
import MinkowskiEngine as ME

if __name__ == "__main__":
    settings.init()

    outer_loop, inner_loop = 1, 1
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset()
    train_loader = DataLoader(dst_train, batch_size=settings.batch_size, shuffle=True, collate_fn=minkowski_collate_fn)
    network = MEConv(in_channel=3, out_channel=10).to(settings.device)
    
    for batch in train_loader:
        input = create_input_batch(batch, True, device=settings.device)
        loss = network(input)
        print(loss)
        

