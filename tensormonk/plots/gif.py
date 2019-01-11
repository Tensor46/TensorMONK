""" TensorMONK :: plots """

import imageio


def make_gif(image_list, gif_name):
    r"""Makes a gif using a list of images.
    """
    if not gif_name.endswith(".gif"):
        gif_name += ".gif"
    imageio.mimsave(gif_name, [imageio.imread(x) for x in image_list])
