import torch
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.renderer import FoVPerspectiveCameras, PointLights, RasterizationSettings, look_at_view_transform, TexturesVertex
from pytorch3d.structures import Meshes
import pytorch3d.io

import matplotlib.pyplot as plt


if __name__=="__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    SHAPENET_PATH = "/home/ghandour/Dataset/ShapeNetCore.v2"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)



    ## Dataset is shufflable

    print(len(shapenet_dataset)) # 52472
    print(shapenet_dataset[0]["label"]) 
    print(device)


    ## Save figure 2D
    R, T = look_at_view_transform(1.0, 1.0, 90)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=512, cull_backfaces=True)
    lights = PointLights(
        device=device, 
        location=[[0.0, 5.0, -10.0]], 
        diffuse_color=((0, 0, 0),),
        specular_color=((0, 0, 0),),
    )

    # 33b1973524b81c4ecafb6b9dbe847b85 model id is a moto with enough details
    images_by_idxs = shapenet_dataset.render(
        model_ids=['33b1973524b81c4ecafb6b9dbe847b85'],
        device=device,
        cameras=cameras,
        raster_settings=raster_settings,
        lights=lights,
        
    )

    ## Create Mesh
    shapenet_model = shapenet_dataset[0]
    model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]
    model_textures = TexturesVertex(verts_features=torch.ones_like(model_verts, device=device)[None])
    shapenet_model_mesh = Meshes(
        verts=[model_verts.to(device)],   
        faces=[model_faces.to(device)],
        textures=model_textures
    )

    ## Save mesh
    pytorch3d.io.IO().save_mesh(shapenet_model_mesh, "final_model.ply", colors_as_uint8=True)

    plt.imshow(images_by_idxs.cpu().numpy()[0, :, :, :3])
    plt.savefig("moto")