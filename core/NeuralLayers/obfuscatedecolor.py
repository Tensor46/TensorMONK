""" TensorMONK's :: NeuralLayers :: ObfuscateDecolor                         """

__all__ = ["ObfuscateDecolor", ]

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
# ============================================================================ #


class ObfuscateDecolor(nn.Module):
    """

        Non-trainable layer that randomly converts the color image to grey and
        obfuscates parts of the image with noise

        * requires performance improvements

        Parameters
            tensor_size :: size of input tensor to pre compute random obfuscations
            p_decolor :: probability of converting rgb to grey
            p_obfuscate :: probability of obfuscating an image
            max_side_obfuscation :: max percentage of height and width to be
                                    obfuscated

    """
    def __init__(self, tensor_size=(1, 3, 60, 40), p_decolor=0.3, p_obfuscate=0.3,
                 max_side_obfuscation=0.2, *args, **kwargs):
        super(ObfuscateDecolor, self).__init__()

        # precomputing random values
        self.p_decolor = torch.rand(600) < p_decolor
        self.p_obfuscate = torch.rand(601) < p_obfuscate

        self.n_decolor = self.p_decolor.numel()
        self.n_obfuscate = self.p_obfuscate.numel()

        self.grey_code = torch.Tensor([[[[0.2126]], [[0.7152]], [[0.0722]]]])

        # precomputing random crops to be ofuscated
        height_obfuscation = []
        for _ in range(646):
            random_start = random.randint(0, int(tensor_size[2]*max_side_obfuscation))
            random_end = random.randint(random_start+1, min(tensor_size[2], random_start + int(tensor_size[2]*max_side_obfuscation)))
            height_obfuscation.append((random_start, random_end))

        width_obfuscation = []
        for i in range(601):
            random_start = random.randint(0, int(tensor_size[3]*max_side_obfuscation))
            random_end = random.randint(random_start+1, min(tensor_size[3], random_start + int(tensor_size[3]*max_side_obfuscation)))
            width_obfuscation.append((random_start, random_end))

        self.sample_height = np.array(height_obfuscation)
        self.sample_width = np.array(width_obfuscation)
        self.n_sample_height = self.sample_height.shape[0]
        self.n_sample_width = self.sample_width.shape[0]

    def forward(self, tensor):

        if tensor.size(1) == 3: # only for RGB images
            for i in range(tensor.size(0)):
                if self.n_decolor == self.p_decolor.size(0):
                    self.n_decolor = 0
                if self.p_decolor[self.n_decolor] == 1:
                    tensor[i,] = F.conv2d(tensor[i,].unsqueeze(0), self.grey_code.cuda() if tensor.is_cuda else self.grey_code).expand(-1, 3, -1, -1)
                self.n_decolor += 1

        random_shit = torch.rand(1, *tensor.size()[1:])
        if tensor.is_cuda:
            random_shit = random_shit.cuda()

        for i in range(tensor.size(0)):
            if self.n_obfuscate == self.p_obfuscate.size(0):
                self.n_obfuscate = 0

            if self.p_obfuscate[self.n_obfuscate] == 1:
                if self.n_sample_height == self.sample_height.shape[0]:
                    self.n_sample_height = 0
                if self.n_sample_width == self.sample_width.shape[0]:
                    self.n_sample_width = 0

                sh, eh = self.sample_height[self.n_sample_height]
                sw, ew = self.sample_width[self.n_sample_width]
                tensor[i, :, sh:eh, sw:ew] = random_shit[0, :, sh:eh, sw:ew]

                self.n_sample_height += 1
                self.n_sample_width += 1
            self.n_obfuscate += 1

        return tensor.detach()


# from PIL import Image as ImPIL
# Image = "./data/test.jpeg"
# Image = ImPIL.open(Image).resize((64,64))
# tensor = (np.array(Image).astype(np.float32).transpose(2, 0, 1)/255.)[np.newaxis,]
# tensor_size = (1, 3, 64, 64)
# test = ObfuscateDecolor(tensor_size, .5, .5)
# ImPIL.fromarray((np.array(test(torch.from_numpy(tensor.copy()))[0,]).transpose(1, 2, 0)*255.).astype(np.uint8))
# for _ in range(10000):
#     test(torch.from_numpy(tensor.copy())).size()
