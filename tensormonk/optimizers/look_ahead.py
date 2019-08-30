""" TensorMONK :: optimizers """

import torch


class LookAhead(torch.optim.Optimizer):
    r"""LookAhead optimizer - https://arxiv.org/pdf/1907.08610.pdf

    Args:
        params (required, str): parameters or parameter groups
        optimizer (optional, torch.nn.Optimizer): Any pytorch optimizer or
            custom version. Default = torch.optim.SGD
        k (optional, int): k fast weight steps, default = 6
        alpha (optional, float): slow weights learning rate, default = 0.5
        kwargs (all the optimizer kwargs): {"lr": 0.1}

    Ex:
        model = torch.nn.Linear(6, 6)
        optimizer = LookAhead(params=model.parameters(),
                              optimizer=torch.optim.SGD,
                              k=6,
                              alpha=0.5,
                              lr=0.1,
                              momemtum=0.9)
    """
    def __init__(self,
                 params,
                 optimizer=torch.optim.SGD,
                 k: int = 6,
                 alpha: float = 0.5,
                 **kwargs):
        if not isinstance(k, int):
            raise TypeError("LookAhead: k must be int")
        if k < 2:
            raise ValueError("LookAhead: k (steps) must be >= 2")
        if not isinstance(alpha, float):
            raise TypeError("LookAhead: alpha must be float")
        if not (0 < alpha < 1):
            raise ValueError("LookAhead: alpha must be > 0 and < 1")

        _optimizer = optimizer(params, **kwargs)
        # parameters for slow weights
        params_clone = []
        for group in _optimizer.param_groups:
            for p in group["params"]:
                p_clone = p.clone().detach()
                p_clone.requires_grad = False
                params_clone.append(p_clone)

        super(LookAhead, self).__init__(params_clone, {})
        self.k = k
        self.alpha = alpha
        self.optimizer = _optimizer
        self.counter = 0

    def __setstate__(self, state):
        super(LookAhead, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        loss = self.optimizer.step()
        self.counter += 1
        if self.counter % self.k == 0:
            # update every k steps
            for slow_group, fast_group in zip(self.param_groups,
                                              self.optimizer.param_groups):
                for sp, fp in zip(slow_group["params"], fast_group["params"]):
                    if fp.grad is None:
                        continue
                    sp.data.add_(self.alpha * (fp.data - sp.data))
                    fp.data.copy_(sp.data)
        return loss
