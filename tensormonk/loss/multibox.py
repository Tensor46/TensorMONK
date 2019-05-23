""" TensorMONK's :: loss :: MultiBoxLoss """

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    r"""
    Training objective of Single Shot MultiBox Detector (SSD)
    Original paper -- https://arxiv.org/pdf/1512.02325.pdf

    Args:
        neg_pos_ratio (float): negatives to positive ratio for hard negative
            mining, default=3
        alpha (float): multiplier for box_loss (loss-loc, eq 1), default=1
    """

    def __init__(self,
                 translator: nn.Module,
                 neg_pos_ratio: float = 3.,
                 alpha: float = 1.,
                 **kwargs):
        super(MultiBoxLoss, self).__init__()

        self.translator = translator
        self.ratio = neg_pos_ratio
        self.alpha = alpha
        self.tensor_size = 1,

    def forward(self,
                gcxcywh_boxes: torch.Tensor,
                predictions: torch.Tensor,
                target_boxes: list, targets: list):
        """
        gcxcywh_boxes - bounding boxes from the network
        predictions - label predictions from the network without softmax
        target_boxes - actual bounding boxes
        targets - actual labels
        """

        # samples, boxes, labels
        ns, nb, nl = predictions.shape

        if self.translator is not None:
            # normalized ltrb_boxes and labels are encoded to priors
            with torch.no_grad():
                _boxes, _targets = [], []
                for x, y in zip(target_boxes, targets):
                    x, y = self.translator(x, y)
                    _boxes.append(x)
                    _targets.append(y)
                target_gcxcywh_boxes = torch.stack(_boxes, 0).detach()
                targets = torch.stack(_targets, 0).detach()
        else:
            target_gcxcywh_boxes, targets = target_boxes, targets

        # Assuming background label is 0 & target boxes are thresholded by
        # iou_threshold - default=0.5
        #       Compute location loss - only considers non-background
        mask_objects = (targets.detach() > 0).view(-1)
        # eq 2 in https://arxiv.org/pdf/1512.02325.pdf (loss loc)
        box_loss = F.smooth_l1_loss(
            gcxcywh_boxes.view(-1, 4).contiguous()[mask_objects, :],
            target_gcxcywh_boxes.view(-1, 4).contiguous()[mask_objects, :],
            reduction="mean")

        # cross entropy with hard negative mining
        mask_objects = mask_objects.view(ns, -1)
        background_probs = F.softmax(predictions.detach(), dim=2)[:, :, 0]
        # Mark the mask_background with hard negatives
        mask_background = torch.zeros(ns, nb).to(targets.device).byte()
        for i in range(ns):
            probs = background_probs[i]
            # Push the object prob to maximum
            probs[mask_objects[i]] = 1
            retain = max(1, int(mask_objects[i].sum() * self.ratio))
            mask_background[i, torch.argsort(probs)[:retain]] = 1

        mask = (mask_objects | mask_background).view(-1)
        # eq 3 in https://arxiv.org/pdf/1512.02325.pdf (loss conf)
        detection_loss = F.cross_entropy(
            predictions.view(-1, nl)[mask], targets.view(-1)[mask],
            reduction="mean")
        # eq - 1 in https://arxiv.org/pdf/1512.02325.pdf
        loss = (detection_loss + self.alpha * box_loss)
        return loss


# from tensormonk.utils import SSDUtils
# gcxcywh_boxes = torch.rand(2, 8732, 4)
# predictions = torch.rand(2, 8732, 3)
# target_boxes = (torch.Tensor([[0.1, 0.1, 0.6, 0.9]]),
#                 torch.Tensor([[0.6, 0.8, 0.6, 0.9], [0.2, 0.3, 0.4, 0.6]]))
# targets = (torch.Tensor([0]).long(), torch.Tensor([0, 2]).long())
#
# translator = SSDUtils.Translator(model="SSD300", var1=.1, var2=.2,
#                                  encode_iou_threshold=0.5)
# test = MultiBoxLoss(translator, 3., 1.)
# gcxcywh_boxes.requires_grad = True
# loss = test(gcxcywh_boxes, predictions, target_boxes, targets)
# loss.backward()
