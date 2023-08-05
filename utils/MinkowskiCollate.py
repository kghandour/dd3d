import MinkowskiEngine as ME
import torch

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"].reshape(1) for d in list_data],
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