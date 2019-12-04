""" TensorMONK :: layers :: FeatureFusion """

__all__ = ["FeatureFusion"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activations.activations import maxout


class FeatureFusion(nn.Module):
    r""" To fuse features from same/different scales, all features must have
    same depth!

    Args:
        n_features (int, required): #Features to be fused

        method (str, optional): method logic after resizing all the tensor's
            to match the first tensor in the args using bilinear interpolation.

            options: "sum" | "maxout-cat" | "fast-normalize" | "softmax"
                "sum"            : usual sum
                "maxout-cat"     : concat((maxout(args[0]), maxout(args[1])))
                                   n_features must be 2
                "fast-normalize" : https://arxiv.org/pdf/1911.09070.pdf
                "softmax"        : https://arxiv.org/pdf/1911.09070.pdf
            default: "softmax"

    """
    METHODS = ("fast-normalize", "maxout-cat", "softmax", "sum")

    def __init__(self, n_features: int, method: str = "softmax"):
        super(FeatureFusion, self).__init__()

        assert n_features >= 2 and isinstance(n_features, int)
        assert method in FeatureFusion.METHODS

        self.n_features = n_features
        self.method = method
        if n_features > 2 and method == "maxout-cat":
            import warnings
            warnings.warn("FeatureFusion: n_features must be 2 for method = "
                          "'maxout-cat', switching method to 'sum'")
            self.method = "sum"
        if method in ("fast-normalize", "softmax"):
            self.weight = nn.Parameter(torch.rand(n_features).softmax(-1))

    def forward(self, *args) -> torch.Tensor:
        assert len(args) == self.n_features
        osz = args[0].shape

        if self.method == "sum":
            return sum([self._resize(x, osz) for x in args])
        if self.method == "maxout-cat":
            return torch.cat([maxout(self._resize(x, osz)) for x in args], 1)

        if self.method == "softmax":
            ws = F.softmax(self.weight, 0)
        elif self.method == "fast-normalize":
            ws = F.relu(self.weight)
            ws = ws / (sum(ws) + 0.0001)
        return sum([w * self._resize(x, osz) for w, x in zip(ws, args)])

    def _resize(self, tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
        return tensor if tensor.shape == shape else \
            F.interpolate(tensor, size=shape[2:], mode="bilinear",
                          align_corners=True)

    def __repr__(self):
        return "FeatureFusion: method = {}".format(self.method)
