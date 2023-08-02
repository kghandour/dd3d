from typing import Any, Callable, Optional, Tuple
from torchvision import datasets, transforms
import torch
import configs.settings as settings

def get_dataset():
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    data_path = settings.mnistconfig.get("dataset_dir")
    ### Ignoring transform because I want it to be binary

    transform = transforms.Compose([transforms.ToTensor(), ThresholdTransform(thr_255=127)])
    dst_train = Mnist2D(data_path, train=True, download=True, transform=transform) # no augmentation
    dst_test = Mnist2D(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=settings.num_workers)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
class Mnist2D(datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform= None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = super().__getitem__(index)
        occ_grid = torch.argwhere(image==1)
        return {
            "coordinates":occ_grid.to(torch.float32),
            "features":occ_grid.to(torch.float32),
            "label": label
        }
    
class ThresholdTransform(object): ## https://stackoverflow.com/questions/65979207/applying-a-simple-transformation-to-get-a-binary-image-using-pytorch
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type