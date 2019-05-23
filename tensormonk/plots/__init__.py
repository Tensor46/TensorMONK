""" TensorMONK :: plots """

__all__ = ["make_gif", "VisPlots", "line_plot"]

from .gif import make_gif
from .visplots import VisPlots
from .line import line_plot

del gif, visplots, line
