from pytorch3d.datasets import ShapeNetCore
from torch.utils.data import Dataset, DataLoader
# import numpy as np
import logging
import MinkowskiEngine as ME
import torch


def resample_mesh(faces, vertices, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """

    vec_cross = torch.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = torch.sqrt(torch.sum(vec_cross ** 2, 1))

    n_samples = (torch.sum(face_areas) * density).type(torch.int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = torch.ceil(density * face_areas).type(torch.int)
    floor_num = torch.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = torch.where(n_samples_per_face > 0)[0].type(torch.float)
        floor_indices = torch.multinomial(indices, floor_num, replacement=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = torch.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = torch.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = torch.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
        (1 - torch.sqrt(r[:, 0:1])) * A
        + torch.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + torch.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P


class MeshLoader(Dataset):
    ### ME examples reconstruciton
    def __init__(
        self,
        dataset,
        num_points = 2048,
        resolution = 128
    ):
        Dataset.__init__(self)
        self.dataset = dataset
        self.num_points = num_points
        self.resolution = resolution
        self.cache = [None]*len(dataset)
        self.last_cache_percent = 0
        self.phase = "train"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        label = self.dataset[index]["label"]
        verts = self.dataset[index]["verts"]
        faces = self.dataset[index]["faces"]
        textures = self.dataset[index]["textures"]

        # if index in self.cache:
        #     xyz = self.cache[index]
        # else:
        xyz = resample_mesh(vertices=verts, faces=faces)

        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        # self.cache[index] = xyz
        # cache_percent = int((len(self.cache) / len(self)) * 100)
        # if (
        #     cache_percent > 0
        #     and cache_percent % 10 == 0
        #     and cache_percent != self.last_cache_percent
        # ):
        #     logging.info(
        #         f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
        #     )
        #     self.last_cache_percent = cache_percent
    
        # Use color or other features if available
        feats = torch.ones((len(xyz), 1))

        # if len(xyz) < 1000:
        #     logging.info(
        #         f"Skipping {self.dataset[index]['model_id']}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
        #     )
        #     return None

        xyz = xyz * self.resolution
        # coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
        # print("coordinates length after", len(coords))
        return {
            "coordinates":xyz,
            "label":label,
            "features":feats,
        }
