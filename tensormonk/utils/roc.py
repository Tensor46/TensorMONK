""" TensorMONK :: utils """

import torch
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def roc(genuine_or_scorematrix, impostor_or_labels, filename=None,
        print_show=False, semilog=True, lower_triangle=True):
    r"""Computes receiver under operating curve for a given combination of
    (genuine and impostor) or (score matrix and labels).

    Args:
        genuine_or_scorematrix: genuine scores or all scores (square matrix) in
            list/tuple/numpy.ndarray/torch.Tensor
        impostor_or_labels: impostor scores or labels in
            list/tuple/numpy.ndarray/torch.Tensor
            list/tuple of strings for labels is accepted
        filename: fullpath of image to save
        print_show: True = prints gars at fars and shows the roc
        semilog: True = plots the roc on semilog
        lower_triangle: True = avoids duplicates in score matrix

    Return:
        A dictionary with gar and their corresponding far, auc, and
        gar_samples.
            gar - genuine accept rates with a range 0 to 1
            far - false accept rates with a range 0 to 1
            auc - area under curve
            gar_samples - gar's at far = 0.00001, 0.0001, 0.001, 0.01, 0.01, 1.
    """
    # convert to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, list) or isinstance(x, tuple):
            assert type(x[0]) in (int, float, str), \
                ("list/tuple of int/float/str are accepted," +
                 " given {}").format(type(x[0]))
            if isinstance(x[0], str):
                classes = sorted(list(set(x)))
                x = [classes.index(y) for y in x]
            return np.array(x)
        else:
            raise NotImplementedError
    gs = to_numpy(genuine_or_scorematrix)
    il = to_numpy(impostor_or_labels)

    # get genuine and impostor scores if score matrix and labels are provided
    if gs.ndim == 2:
        if gs.shape[0] == gs.shape[1] and gs.shape[0] == il.size:
            # genuine_or_scorematrix is a score matrix
            if lower_triangle:
                indices = il.reshape((-1, 1))
                indices = np.concatenate([indices]*indices.shape[0], 1)
                indices = (indices == indices.T).astype(np.int) + 1
                indices = np.tril(indices, -1).flatten()
                genuine = gs.flatten()[indices == 2]
                impostor = gs.flatten()[indices == 1]
            else:
                indices = np.expand_dims(il, 1) == np.expand_dims(il, 0)
                genuine = gs.flatten()[indices.flatten()]
                indices = np.expand_dims(il, 1) != np.expand_dims(il, 0)
                impostor = gs.flatten()[indices.flatten()]
    if "genuine" not in locals():
        # genuine_or_scorematrix is an array of genuine scores
        genuine = gs.flatten()
        impostor = il.flatten()

    # convert to float32
    genuine, impostor = genuine.astype(np.float32), impostor.astype(np.float32)
    # min and max
    min_score = min(genuine.min(), impostor.min())
    max_score = max(genuine.max(), impostor.max())
    # find histogram bins and then count
    bins = np.arange(min_score, max_score, (max_score-min_score)/4646)
    genuine_bin_count = np.histogram(genuine, density=False, bins=bins)[0]
    impostor_bin_count = np.histogram(impostor, density=False, bins=bins)[0]
    genuine_bin_count = genuine_bin_count.astype(np.float32) / genuine.size
    impostor_bin_count = impostor_bin_count.astype(np.float32) / impostor.size
    was_distance = False
    if genuine.mean() < impostor.mean():  # distance bins to similarity bins
        genuine_bin_count = genuine_bin_count[::-1]
        impostor_bin_count = impostor_bin_count[::-1]
        was_distance = True
    # compute frr & grr, then far = 100 - grr & gar = 100 - frr
    gar = 1 - (1. * np.cumsum(genuine_bin_count))
    far = 1 - (1. * np.cumsum(impostor_bin_count))
    # Find gars on log scale -- 0.00001 - 1
    far_samples = [10**x for x in range(-5, 1)]
    gar_samples, thr_samples = [], []
    for x in far_samples:
        try:
            idx = np.where(abs(far - x).clip(1e-6) == 1e-6)[0].max() \
                if x == 1 else np.argmin(np.abs(far - x))
            gar_samples.append(round(gar[idx], 6))
            thr_samples.append(round(bins[(-idx) if was_distance else idx], 6))
        except IndexError:
            # when accurate far's cannot be estimated -- precision issues
            gar_samples.append(np.nan)
            thr_samples.append(np.nan)
    samples = [gar[np.argmin(np.abs(far - 10**x))] for x in range(-5, 1)]
    if print_show:
        print("reference fars :: 1e-05/1e-04/0.001/0.010/0.100/1.000")
        print(("-^- their gars :: " +
              "/".join(["{:1.3f}"]*6)).format(*gar_samples))
        print(("-^- their thrs :: " +
              "/".join(["{:1.3f}"]*6)).format(*thr_samples))
    # interpolate and shirnk gar & far to 600 samples, for ploting
    _gar = interp.interp1d(np.arange(gar.size), gar)
    gar = _gar(np.linspace(0, gar.size-1, 599))
    _far = interp.interp1d(np.arange(far.size), far)
    far = _far(np.linspace(0, far.size-1, 599))

    gar = np.concatenate((np.array([1.]), gar), axis=0)
    far = np.concatenate((np.array([1.]), far), axis=0)

    if filename is not None:
        if not filename.endswith((".png", ".jpeg", "jpg")):
            filename += ".png"
        # TODO seaborn ?
        plt.semilogx(far, gar)
        plt.xlabel("far")
        plt.ylabel("gar")
        plt.ylim((-0.01, 1.01))
        plt.savefig(filename, dpi=300)
        if print_show:
            plt.show()

    return {"gar": gar, "far": far, "auc": abs(np.trapz(gar, far)),
            "gar_samples": gar_samples, "far_samples": far_samples,
            "thr_samples": thr_samples}
