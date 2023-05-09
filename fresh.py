from pytorch3d.datasets import ShapeNetCore
import configparser

if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    def_conf = config["DEFAULT"]
    shnet = ShapeNetCore(def_conf.get("shapenet_path", "/home/ghandour/Dataset/ShapeNetCore.v2"))

    

