""" TensorMONK's :: essentials """

import torch
import numpy as np
from ..loss.utils import compute_top15


class Meter(object):
    r"""A meter that accumulates all the values. Requires scalar Tensor or
    float. Rounds the output to ndigits.
    """
    def __init__(self, ndigits: int = 2):
        self.values = []
        self.ndigits = ndigits

    def update(self, current: torch.Tensor) -> None:
        if isinstance(current, torch.Tensor):
            current = float(current.detach().mean().cpu().numpy())
            current = round(current, self.ndigits)
        self.values.append(current)
        return

    def average(self, n: int = 1000) -> float:
        if len(self.values) == -1:
            if len(self.values) == 0:
                avg = 0.
            else:
                avg = np.mean(self.values)
        elif 0 <= len(self.values) < n:
            if len(self.values) == 0:
                avg = 0.
            else:
                avg = np.mean(self.values)
        else:
            avg = np.mean(self.values[-n:])
        return round(avg, self.ndigits)

    def reset(self):
        self.values = []


class AverageMeter(object):
    def __init__(self, ndigits: int = 2):
        self.value = 0
        self.n = 0
        self.ndigits = ndigits

    def update(self, current: float) -> None:
        self.value += current
        self.n += 1
        return

    def average(self) -> float:
        if self.n == 0:
            return 0.
        return round(self.value / self.n, self.ndigits)

    def reset(self):
        self.value = 0
        self.n = 0


class AccuracyMeter(object):
    def __init__(self, ndigits: int = 2):
        self.correct = 0
        self.total = 0
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.ndigits = ndigits

    def update(self, probability: torch.Tensor, targets: torch.Tensor):
        targets = targets.long().to(probability.device)
        top1, top5 = compute_top15(probability, targets)
        self.top1.update(top1)
        self.top5.update(top5)

        predicted = probability.max(1)[1]
        predicted, targets = predicted.view(-1).long(), targets.view(-1)
        self.correct += float((predicted == targets).sum().data.numpy())
        self.total += targets.numel()
        return

    def top15(self) -> (float, float):
        if self.total == 0:
            return 0., 0.
        return round(self.top1.average() * 100., self.ndigits), \
            round(self.top5.average() * 100., self.ndigits)

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.
        return round(100. * self.correct / self.total, self.ndigits)

    def error(self) -> float:
        if self.total == 0:
            return 100.
        return round(100. - self.accuracy(), self.ndigits)

    def reset(self):
        self.correct = 0
        self.total = 0
        self.top1.reset()
        self.top5.reset()
