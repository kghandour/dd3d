import numpy as np
from scipy.linalg import expm, norm
import random

class RandomRotation:

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        R = self._M(
            np.random.rand(3) - 0.5, 2 * np.pi * (np.random.rand(1) - 0.5))
        return coords @ R, feats


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, feats):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s, feats


class RandomShear:

    def __call__(self, coords, feats):
        T = np.eye(3) + np.random.randn(3, 3)
        return coords @ T, feats


class RandomTranslation:

    def __call__(self, coords, feats):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans, feats
    

class CoordinateTranslation:
    def __init__(self, translation):
        self.trans = translation

    def __call__(self, coords):
        if self.trans > 0:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        return coords
    
class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords *= np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords += np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords

    def __repr__(self):
        return f"Transformation(scale={self.scale_range}, translation={self.trans}, jitter={self.jitter})"