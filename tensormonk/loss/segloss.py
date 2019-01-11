""" TensorMONK :: loss :: segloss """

import torch


class DiceLoss(torch.nn.Module):
    r""" Dice/ Tversky loss for semantic segmentationself.
    Implemented from https://arxiv.org/pdf/1803.11078.pdf
    https://arxiv.org/pdf/1706.05721.pdf has same equation but with alpha
    and beta controlling FP and FN.
    Args:
        type: tversky/dice
    Definations:
        p_i - correctly predicted foreground pixels
        p_j - correctly predicted background pixels
        g_i - target foreground pixels
        g_j - target background pixels
        p_i * g_i - True Positives  (TP)
        p_i * g_j - False Positives (FP)
        p_j * g_i - False Negatives (FN)
    """
    def __init__(self, type="tversky", *args, **kwargs):
        super(DiceLoss, self).__init__()
        self.tensor_size = (1,)
        if type == "tversky":
            self.beta = 2.0
            # for https://arxiv.org/pdf/1706.05721.pdf,
            # beta of 2 results in alpha of 0.2 and beta of 0.8
        elif type == "dice":
            self.beta = 1.0         # below Eq(6)
        else:
            raise NotImplementedError

    def forward(self, prediction, targets):
        top1, top5 = 0., 0.
        if prediction.shape[1] == 1:
            p_i = prediction
            p_j = prediction.mul(-1).add(1)
            g_i = targets
            g_j = targets.mul(-1).add(1)
            # the above is similar to one hot encoding of targets
            num = (p_i*g_i).sum(1).sum(1).mul((1 + self.beta**2))   # eq(5)
            den = num.add((p_i*g_j).sum(1).sum(1).mul((self.beta**2))) \
                .add((p_j*g_i).sum(1).sum(1).mul((self.beta)))    # eq(5)
            loss = num / den.add(1e-6)
        elif prediction.shape[1] == 2:
            p_i = prediction[:, 0, :, :]
            p_j = prediction[:, 1, :, :]
            g_i = targets
            g_j = targets.mul(-1).add(1)
            # the above is similar to one hot encoding of targets
            num = (p_i*g_i).sum(1).sum(1).mul((1 + self.beta**2))   # eq(5)
            den = num.add((p_i*g_j).sum(1).sum(1).mul((self.beta**2))) \
                .add((p_j*g_i).sum(1).sum(1).mul((self.beta)))    # eq(5)
            loss = num / den.add(1e-6)
        else:
            raise NotImplementedError
        return loss.mean(), (top1, top5)
