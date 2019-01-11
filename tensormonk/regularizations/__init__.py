""" TensorMONK :: regularizations """

__all__ = ["DropOut"]


def DropOut(tensor_size, p, dropblock=True, **kwargs):
    import torch.nn as nn
    if p > 0:
        if len(tensor_size) == 4:
            if dropblock:
                from .dropblock import DropBlock
                kwgs = {}
                if "block_size" in kwargs.keys():
                    kwgs["block_size"] = kwargs["block_size"]
                if "shared" in kwargs.keys():
                    kwgs["shared"] = kwargs["shared"]
                if "iterative_p" in kwargs.keys():
                    kwgs["iterative_p"] = kwargs["iterative_p"]
                if "steps_to_max" in kwargs.keys():
                    kwgs["steps_to_max"] = kwargs["steps_to_max"]
                return DropBlock(tensor_size, p=p, **kwgs)
            else:
                return nn.Dropout2d(p)
        else:
            return nn.Dropout(p)
    else:
        return None
