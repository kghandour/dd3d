import MinkowskiEngine as ME
import torch
import pytorch3d
from pytorch3d.datasets import ShapeNetCore
from torch.utils.data import Dataset, DataLoader
from temp_py.meshLoader import MeshLoader
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

def stack_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = (
        torch.stack([d["coordinates"] for d in list_data]),
        torch.stack([d["features"] for d in list_data]),
        torch.cat([d["label"] for d in list_data]),
    )

    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }



if __name__=="__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        
    SHAPENET_PATH = "/home/ghandour/Dataset/ShapeNetCore.v2"
    shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=2)

    num_points=2048

    print(shapenet_dataset[0].keys())
    pcd = MeshLoader(shapenet_dataset)

    ## Create Mesh
    shapenet_model = shapenet_dataset[0]
    model_verts, model_faces = shapenet_model["verts"], shapenet_model["faces"]
    model_textures = TexturesVertex(verts_features=torch.ones_like(model_verts, device=device)[None])
    shapenet_model_mesh = Meshes(
        verts=[model_verts.to(device)],   
        faces=[model_faces.to(device)],
        textures=model_textures
    )

    IO().save_mesh(shapenet_model_mesh, "output_mesh.obj")

    shapenet_loader = DataLoader(pcd, batch_size=1, collate_fn=minkowski_collate_fn)
    print(len(shapenet_loader))
    it = iter(shapenet_loader)
    shapenet_batch = next(it)
    batch_renderings = shapenet_batch
    pcl = Pointclouds([pcd[0]["coordinates"]])
    IO().save_pointcloud(pcl, "output_pcd.ply")
    # pointcloud = Pointclouds(batch_renderings["coordinates"])
