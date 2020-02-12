"""TensorMONK :: loss :: AdversarialLoss"""

__all__ = ["AdversarialLoss"]

import torch
import numpy as np
eps = np.finfo(float).eps


def g_minimax(d_of_fake: torch.Tensor, invert_labels: bool = False):
    r"""Minimax loss for generator. (`"Generative Adversarial Nets"
    <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`_).
    Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)).

    loss = - log(σ( d_of_fake ))

    Args:
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    if invert_labels:
        return - (1 - d_of_fake.sigmoid()).clamp(eps).log().mean()
    return - d_of_fake.sigmoid().clamp(eps).log().mean()


def d_minimax(d_of_real: torch.Tensor, d_of_fake: torch.Tensor,
              invert_labels: bool = False):
    r"""Minimax loss for discriminator. (`"Generative Adversarial Nets"
    <https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>`_).
    Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)) and d_of_real = D(I).

    loss = - log(σ( d_of_real )) - log(1 - σ( d_of_fake ))

    Args:
        d_of_real (torch.Tensor, required): discriminator output of real.
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_real, torch.Tensor)
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    if invert_labels:
        rloss = - (1 - d_of_real.sigmoid()).clamp(eps).log()
        floss = - d_of_fake.sigmoid().clamp(eps).log()
    else:
        rloss = - d_of_real.sigmoid().clamp(eps).log()
        floss = - (1 - d_of_fake.sigmoid()).clamp(eps).log()
    return (rloss + floss).mean() / 2


def g_least_squares(d_of_fake: torch.Tensor, invert_labels: bool = False):
    r"""Least squares loss for generator. (`"Least Squares Generative
    Adversarial Networks"<https://arxiv.org/abs/1611.04076>`_).
    Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)).

    loss = (1 - σ( d_of_fake ))^2

    Args:
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    if invert_labels:
        return d_of_fake.sigmoid().pow(2).mean()
    return (1 - d_of_fake.sigmoid()).pow(2).mean()


def d_least_squares(d_of_real: torch.Tensor, d_of_fake: torch.Tensor,
                    invert_labels: bool = False):
    r"""Least squares loss for generator. (`"Least Squares Generative
    Adversarial Networks"<https://arxiv.org/abs/1611.04076>`_).
    Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)) and d_of_real = D(I).

    loss = ((1 - σ( d_of_real ))^2 + σ( d_of_fake )^2) / 2

    Args:
        d_of_real (torch.Tensor, required): discriminator output of real.
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_real, torch.Tensor)
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    if invert_labels:
        rloss = d_of_real.sigmoid().pow(2)
        floss = (1 - d_of_fake.sigmoid()).pow(2)
    else:
        rloss = (1 - d_of_real.sigmoid()).pow(2)
        floss = d_of_fake.sigmoid().pow(2)
    return (rloss + floss).mean() / 2


def g_relativistic(d_of_real: torch.Tensor, d_of_fake: torch.Tensor,
                   invert_labels: bool = False):
    r"""Relativistic loss for generator. (`"The relativistic discriminator: a
    key element missing from standard GAN"<https://arxiv.org/abs/1807.00734>`_
    ). Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)) and d_of_real = D(I).

    loss = - log(1 - σ(d_of_fake - E[d_of_real]))

    Args:
        d_of_real (torch.Tensor, required): discriminator output of real.
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_real, torch.Tensor)
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    if invert_labels:
        return - (d_of_fake - d_of_real.mean()).sigmoid().clamp(eps).log()
    return - (1 - (d_of_fake - d_of_real.mean()).sigmoid()).clamp(eps).log()


def d_relativistic(d_of_real: torch.Tensor, d_of_fake: torch.Tensor,
                   invert_labels: bool = False):
    r"""Relativistic loss for generator. (`"The relativistic discriminator: a
    key element missing from standard GAN"<https://arxiv.org/abs/1807.00734>`_
    ). Assumes, real label is 1 and fake label is 0 (use invert_labels to flip
    real and fake labels). d_of_fake = D(G(z)) and d_of_real = D(I).

    loss = - log(1 - σ(d_of_real - E[d_of_fake])) -
        log(σ(d_of_fake - E[d_of_real]))

    Args:
        d_of_real (torch.Tensor, required): discriminator output of real.
        d_of_fake (torch.Tensor, required): discriminator output of fake.
        invert_labels (bool, optional): Inverts real and fake labels to 0 and 1
            respectively (default: :obj:`"False"`).
    """
    assert isinstance(d_of_real, torch.Tensor)
    assert isinstance(d_of_fake, torch.Tensor)
    assert isinstance(invert_labels, bool)
    dra_rf = (d_of_real - d_of_fake.mean()).sigmoid().clamp(eps)
    dra_fr = (d_of_fake - d_of_real.mean()).sigmoid().clamp(eps)
    if invert_labels:
        return - (dra_rf.log() + (1 - dra_fr).log()).mean()
    return - ((1 - dra_rf).log() + dra_fr.log()).mean()


class AdversarialLoss:
    r"""Adversarial losses.

    Assumes 1 is real and 0 is fake.
        Fake --> D(G(z)) = d_of_fake = d_of_g_of_z
        Real --> D(I) = d_of_real
    """

    # Paper: Generative Adversarial Nets
    # URL:   https://arxiv.org/abs/1406.2661
    g_minimax = g_minimax
    d_minimax = d_minimax
    # Paper: Least Squares Generative Adversarial Networks
    # URL:   https://arxiv.org/abs/1611.04076
    g_least_squares = g_least_squares
    d_least_squares = d_least_squares
    # Paper: The relativistic discriminatorr: a key element missing from
    #        standard GAN
    # URL:   https://arxiv.org/pdf/1807.00734.pdf
    g_relativistic = g_relativistic
    d_relativistic = d_relativistic
