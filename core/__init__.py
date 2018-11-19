""" TensorMONK's                                                             """

from . import NeuralArchitectures, NeuralEssentials, NeuralLayers


class _utils:

    @staticmethod
    def corr_1d(tensor_a, tensor_b):
        assert tensor_a.dim() == 2 and tensor_b.dim() == 2, \
            "correlation_1d :: tensor_a and tensor_b must be 2D"

        return (tensor_a.mul(tensor_b).mean(1) - tensor_a.mean(1)*tensor_b.mean(1))/\
            ((tensor_a.pow(2).mean(1) - tensor_a.mean(1).pow(2)).pow(0.5) *
             (tensor_b.pow(2).mean(1) - tensor_b.mean(1).pow(2)).pow(0.5))

    @staticmethod
    def xcorr_1d(tensor):
        assert tensor.dim() == 2, "xcorr_1d :: tensor must be 2D"
        n = tensor.size(0)
        return (tensor.view(n, 1, -1).mul(tensor.view(1, n, -1)).mean(2)
            - tensor.view(n, 1, -1).mean(2).mul(tensor.view(1, n, -1).mean(2))) / \
            ((tensor.view(n, 1, -1).pow(2).mean(2) -
              tensor.view(n, 1, -1).mean(2).pow(2)).pow(0.5) *
             (tensor.view(1, n, -1).pow(2).mean(2) -
              tensor.view(1, n, -1).mean(2).pow(2)).pow(0.5))


utils = _utils()
del _utils
