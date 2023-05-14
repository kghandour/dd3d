from pytorch3d.datasets import ShapeNetCore
import configparser
import torch
from model.me_network import MinkowskiFCNN
from model.me_classification import train, test

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    def_conf = config["DEFAULT"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===================ModelNet40 Dataset===================")
    print(f"Training with translation", def_conf.get("train_translation"))
    print(f"Evaluating with translation", def_conf.get("train_translation"))
    print("=============================================\n\n")

    net = MinkowskiFCNN(
        in_channel=3, out_channel=2, embedding_channel=1024
    ).to(device)
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")

    train(net, device, config)
    accuracy = test(net, device, config, phase="test")
    print(f"Test accuracy: {accuracy}")
    

