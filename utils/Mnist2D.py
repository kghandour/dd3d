from typing import Any, Callable, Optional, Tuple
from torchvision import datasets, transforms
import torch
import configs.settings as settings
from torch.utils.data import Dataset, DataLoader

from utils.MinkowskiCollate import minkowski_collate_fn

def get_dataset(pixel_val = False):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    data_path = settings.mnistconfig.get("dataset_dir")
    ### Ignoring transform because I want it to be binary
    if(pixel_val):
      transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    else:
      transform = transforms.Compose([transforms.ToTensor(), ThresholdTransform(thr_255=75)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=settings.num_workers)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
class Mnist2D(datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform= None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        occ_grid = torch.argwhere(image==1)[:settings.num_points]
        return {
            "coordinates":occ_grid.to(torch.float32),
            "features":occ_grid.to(torch.float32),
            "label": label
        }
    
    def __len__(self) -> int:
       return super().__len__()
    
class Mnist2Dreal(Dataset):
  def __init__(self, images_tensor, labels_tensor, pixel_val=False, fixed_pt=False):
    self.images = images_tensor
    self.labels = labels_tensor
    self.pixel_val = pixel_val
    self.fixed_pt = fixed_pt

  def __getitem__(self, index):
    image = self.images[index]
    label = self.labels[index]
    if not self.pixel_val and not self.fixed_pt:
      occ_grid = torch.argwhere(image==1)[:settings.num_points]
      return {
          "coordinates":occ_grid.to(torch.float32),
          "features":occ_grid.to(torch.float32),
          "label": label.to(torch.int64)
      }
    elif self.fixed_pt and not self.pixel_val:
      return {
          "coordinates": image.to(torch.float32),
          "features":image.to(torch.float32),
          "label": label.to(torch.int64)
      } 
    flattened = torch.flatten(image)
    unsqueezed = torch.unsqueeze(flattened, axis=1)
    occ_grid = torch.argwhere(image)
    return {
      "coordinates":occ_grid.to(torch.float32),
      "features":unsqueezed.to(torch.float32),
      "label": label.to(torch.int64)
    }
  
  def __len__(self):
     return self.images.shape[0]
  
class Mnist2Dsyn(Dataset):
  def __init__(self, images_tensor, labels_tensor, pixel_val=False):
    self.images = images_tensor
    self.labels = labels_tensor
    self.pixel_val = pixel_val

  def __getitem__(self, index):
    image = self.images[index]
    label = self.labels[index]
    if not self.pixel_val:
      return {
          "coordinates":image.to(torch.float32),
          "features":image.to(torch.float32),
          "label": label.to(torch.int64)
      }
    flattened = image.reshape(-1, 1)
    occ_grid = torch.argwhere(image)
    return {
      "coordinates":occ_grid.to(torch.float32),
      "features":flattened.to(torch.float32),
      "label": label.to(torch.int64)
    }
  
  def __len__(self):
     return self.images.shape[0]
    
  
def get_mnist_dataloader(dataset, batch_size):
  train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=minkowski_collate_fn)
  return train_loader   

class ThresholdTransform(object): ## https://stackoverflow.com/questions/65979207/applying-a-simple-transformation-to-get-a-binary-image-using-pytorch
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]