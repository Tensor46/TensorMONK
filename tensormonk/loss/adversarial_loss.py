""" TensorMONK :: loss :: AdversarialLoss """

__all__ = ["AdversarialLosses"]

import torch


def g_minimax(d_of_fake: torch.Tensor):
    r"""Minimax loss for generator.
    loss = - log(1 - σ( D(G(z)) ))
    """
    return - (1 - d_of_fake.sigmoid()).clamp(1e-8).log().mean()


def d_minimax(d_of_real: torch.Tensor, d_of_fake: torch.Tensor):
    r"""Minimax loss for discriminator.
    loss = - log(1 - σ( D(I) )) - log(σ( D(G(z)) ))
    """
    rloss = - (1 - d_of_real.sigmoid()).clamp(1e-8).log()
    floss = - d_of_fake.sigmoid().clamp(1e-8).log()
    return (rloss + floss).mean() / 2


def g_least_squares(d_of_fake: torch.Tensor):
    r"""Least squares loss for generator.
    loss = (1 - D(G(z)))^2 / 2
    """
    return (1 - d_of_fake).pow(2).mean() / 2


def d_least_squares(d_of_real: torch.Tensor, d_of_fake: torch.Tensor):
    r"""Least squares loss for discriminator.
    loss = (1 - D(I))^2 / 2 + D(G(z))^2 / 2
    """
    rloss = (1 - d_of_real).pow(2) / 2
    floss = d_of_fake.pow(2) / 2
    return (rloss + floss).mean()


def g_relativistic(d_of_real: torch.Tensor, d_of_fake: torch.Tensor):
    r"""ESRGAN equation 2 for generator.
    loss = - log(1 - σ(D(I) - E[D(G(z))])) - log(σ(D(G(z)) - E[D(I)]))
    """
    dra_rf = (d_of_real - d_of_fake.mean()).sigmoid()
    dra_fr = (d_of_fake - d_of_real.mean()).sigmoid()
    return (- (1 - dra_rf).clamp(1e-8).log() - dra_fr.log().clamp(1e-8)).mean()


def d_relativistic(d_of_real: torch.Tensor, d_of_fake: torch.Tensor):
    r"""ESRGAN equation 1 for discriminator.
    loss = - log(σ(D(I) - E[D(G(z))])) - log(1 - σ(D(G(z)) - E[D(I)]))
    """
    dra_rf = (d_of_real - d_of_fake.mean()).sigmoid()
    dra_fr = (d_of_fake - d_of_real.mean()).sigmoid()
    return (- dra_rf.clamp(1e-8).log() - (1 - dra_fr).clamp(1e-8).log()).mean()


class AdversarialLosses:
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
