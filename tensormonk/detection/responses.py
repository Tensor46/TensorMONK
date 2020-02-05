""" TensorMONK's :: detection :: Responses """

__all__ = ["Responses"]
import torch


class Responses:
    r"""An object with all the predictions from anchor detector. The list of
    properties are (:obj:`label`, :obj:`score`, :obj:`boxes`,
    :obj:`point`, :obj:`objectness`, :obj:`centerness`)

    Args:
        label (torch.Tensor/None): Predicted labels.
        score (torch.Tensor/None): Predicted scores.
        boxes (torch.Tensor/None): Predicted boxes (encoded).
        point (torch.Tensor/None): Predicted point (encoded)
        objectness (torch.Tensor/None): Predicted objectness.
        centerness (torch.Tensor/None): Predicted centerness.
    """

    def __init__(self,
                 label: torch.Tensor,
                 score: torch.Tensor,
                 boxes: torch.Tensor,
                 point: torch.Tensor,
                 objectness: torch.Tensor,
                 centerness: torch.Tensor):
        self.label, self.score = label, score
        self.boxes, self.point = boxes, point
        self.objectness, self.centerness = objectness, centerness

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: label value must be None/Tensor")
        self._label = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: score value must be None/Tensor")
        self._score = value

    @property
    def boxes(self):
        return self._boxes

    @boxes.setter
    def boxes(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: boxes value must be None/Tensor")
        self._boxes = value

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: point value must be None/Tensor")
        self._point = value

    @property
    def objectness(self):
        return self._objectness

    @objectness.setter
    def objectness(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: objectness value must be None/Tensor")
        self._objectness = value

    @property
    def centerness(self):
        return self._centerness

    @centerness.setter
    def centerness(self, value: torch.Tensor):
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Responses: centerness value must be None/Tensor")
        self._centerness = value
