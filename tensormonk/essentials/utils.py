""" TensorMONK's :: essentials """

__all__ = ["Meter", "AverageMeter", "AccuracyMeter", "ProgressBar"]

import os
import sys
import time
import datetime
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


class ProgressBar(object):
    r"""Progress bar."""

    def __init__(self, n_iterations: int):
        r"""Initialize progress bar."""
        try:
            _, cols = os.popen('stty size', 'r').read().split()
        except ValueError:
            cols = 60
        self.cols = cols
        self.n_iterations = n_iterations
        self.reset

    @property
    def reset(self):
        r"""Reset progress bar."""
        if hasattr(self, "iteration"):
            if self.iteration > 0:
                print(self.msg)
        self.soft_reset
        return None

    @property
    def soft_reset(self):
        r"""Reset time and iteration."""
        self.iteration = 0
        self.time = []
        self.last = time.time()
        return None

    def __call__(self, add_msg: str = ""):
        r"""Output to monitor."""
        self.iteration += 1
        msg = self.progress() + " " + self.eta()
        if isinstance(add_msg, str) and len(add_msg) > 0:
            msg = msg + " " + add_msg
        self.msg = msg + (" " * 4)
        print(self.msg, end="\r")
        sys.stdout.flush()

    def progress(self) -> str:
        r"""Compute progress."""
        percent = min(100., float(self.iteration) / self.n_iterations * 100)
        percent = "100.0%" if percent == 100 else "{:3.2f}%".format(percent)
        if len(percent) == 5:
            percent = "0" + percent
        return "{" + percent + "}"

    def eta(self) -> str:
        r"""Compute ETA."""
        if self.iteration >= self.n_iterations:
            msg = str(datetime.timedelta(seconds=sum(self.time))).split(".")[0]
            msg = ("0" if len(msg) == 7 else "") + msg
            return "{took " + msg + "}"
        now = time.time()
        self.time.append(now - self.last)
        self.last = now
        avg_seconds = sum(self.time) / len(self.time)
        eta = max(0, (self.n_iterations - self.iteration) * avg_seconds)
        msg = str(datetime.timedelta(seconds=eta)).split(".")[0]
        msg = ("0" if len(msg) == 7 else "") + msg
        if self.iteration == 0:
            return "{eta  --:--:--}"
        return "{eta  " + msg + "}"
