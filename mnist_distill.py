from configs import settings
from utils.MnistDataset import MnistDataset
from torch.utils.data import DataLoader
import utils.MinkowskiCollate as MinkowskiCollate
import MinkowskiEngine as ME

if __name__ == "__main__":
    settings.init()

    outer_loop, inner_loop = 1, 1
    mnist = MnistDataset(phase="train")
    mnist_train = DataLoader(
        mnist, num_workers=settings.num_workers, collate_fn=MinkowskiCollate.minkowski_collate_fn, batch_size=settings.batch_size
    )
    for i, batch in mnist_train:
        input = ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        print("ADD NETWORK HERE")
        pass

