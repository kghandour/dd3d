from pytorch3d.datasets import ShapeNetCore
from torch.utils.data import Dataset, DataLoader
# import numpy as np
import logging
import MinkowskiEngine as ME
import torch


def resample_mesh(faces, vertices, density=1000):
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
        sample_density = 1000,
        quantization_size = 1,
    ):
        Dataset.__init__(self)
        self.dataset = dataset
        self.num_points = num_points
        self.sample_density = sample_density
        self.quantization_size = quantization_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        label = self.dataset[index]["label"]
        verts = self.dataset[index]["verts"]
        faces = self.dataset[index]["faces"]
        textures = self.dataset[index]["textures"]

        xyz = resample_mesh(vertices=verts, faces=faces, density=self.sample_density)

        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]

        feats = torch.ones((len(xyz), 1))

        coords, feats, labels = ME.utils.sparse_quantize(
            coordinates=xyz,
            features=feats,
            labels=label,
            quantization_size=self.sample_density)

        return {
            "coordinates":coords,
            "label":label,
            "features":feats,
            "verts":verts,
            "faces":faces,
        }
