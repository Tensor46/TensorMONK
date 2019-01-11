""" TensorMONK :: layers :: utils """

__all__ = ["check_strides", "check_residue", "update_kwargs"]


def check_strides(strides):
    return (strides > 1 if isinstance(strides, int) else
            (strides[0] > 1 or strides[1] > 1))


def check_residue(strides, t_size, out_channels):
    return check_strides(strides) or t_size[1] != out_channels


def update_kwargs(kwargs, *args):
    if len(args) > 0 and args[0] is not None:
        kwargs["tensor_size"] = args[0]
    if len(args) > 1 and args[1] is not None:
        kwargs["filter_size"] = args[1]
    if len(args) > 2 and args[2] is not None:
        kwargs["out_channels"] = args[2]
    if len(args) > 3 and args[3] is not None:
        kwargs["strides"] = args[3]
    if len(args) > 4 and args[4] is not None:
        kwargs["pad"] = args[4]
    if len(args) > 5 and args[5] is not None:
        kwargs["activation"] = args[5]
    if len(args) > 6 and args[6] is not None:
        kwargs["dropout"] = args[6]
    if len(args) > 7 and args[7] is not None:
        kwargs["normalization"] = args[7]
    if len(args) > 8 and args[8] is not None:
        kwargs["pre_nm"] = args[8]
    if len(args) > 9 and args[9] is not None:
        kwargs["groups"] = args[9]
    if len(args) > 10 and args[10] is not None:
        kwargs["weight_nm"] = args[10]
    if len(args) > 11 and args[11] is not None:
        kwargs["equalized"] = args[11]
    if len(args) > 12 and args[12] is not None:
        kwargs["shift"] = args[12]
    if len(args) > 13 and args[13] is not None:
        kwargs["bias"] = args[13]
    if len(args) > 14 and args[14] is not None:
        kwargs["dropblock"] = args[14]
    return kwargs
