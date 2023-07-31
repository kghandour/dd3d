from configs import settings
from utils.MnistDataset import MnistDataset


if __name__ == "__main__":
    settings.init()

    outer_loop, inner_loop = 1, 1
    mnist = MnistDataset()
    