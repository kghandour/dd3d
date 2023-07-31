from models.ConvNet3D import ConvNet3D
from models.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model, get_loss, get_ce_loss
import configs.settings as settings

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_network(name: str):
    name = str.lower(name)
    if(name=="pointnet" or name=="pointnet++"):
        return get_model(settings.num_classes, False)
    elif(name=="convnet" or name=="convnet3d"):
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        net = ConvNet3D(channel=3, num_classes=settings.num_classes, num_points=settings.num_points, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
        return net