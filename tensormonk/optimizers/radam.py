""" TensorMONK :: optimizers """


import torch


class RAdam(torch.optim.Optimizer):
    r""" On the Variance of the Adaptive Learning Rate and Beyond -
    https://arxiv.org/pdf/1908.03265v1.pdf
    * Built on PyTorch's Adam optimizer

    Args:
        params (required, str): parameters or parameter groups
        lr (optional, float): default = 1e-3
        betas (optional, tuple): default = (0.9, 0.999)
        eps (optional, float): default = 1e-8
        weight_decay (optional, float): default = 0
        n_warmup (optional, int): default = 1000

    Ex:
        model = torch.nn.Linear(6, 6)
        optimizer = RAdam(params=model.parameters())
    """
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.,
                 n_warmup: int = 1000):

        if not isinstance(lr, float):
            raise TypeError("RAdam: lr must be float")
        if not isinstance(betas, (list, tuple)):
            raise TypeError("RAdam: betas must be list/tuple")
        if len(betas) != 2:
            raise ValueError("RAdam: betas must be list/tuple of length 2")
        if not isinstance(eps, float):
            raise TypeError("RAdam: eps must be float")
        if not isinstance(weight_decay, float):
            raise TypeError("RAdam: weight_decay must be float")
        if not isinstance(n_warmup, int):
            raise TypeError("RAdam: n_warmup must be int")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        n_warmup=n_warmup)
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse "
                                       "gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of mt & vt
                    state["mt"] = torch.zeros_like(p.data)
                    state["vt"] = torch.zeros_like(p.data)

                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # update moving average of mt & vt
                mt, vt = state["mt"], state["vt"]
                mt.mul_(beta1).add_(1 - beta1, grad)
                vt.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # bias corrected first moment running average
                c_mt = mt.div(1 - (beta1 ** state["step"]))

                # rho computation
                rho = 2 / (1 - beta2) - 1
                rhot = rho - (2 * state["step"] * (beta2 ** state["step"]) /
                              (1 - (beta2 ** state["step"])))

                # learning rate with warm-up
                alphat = group["lr"] if state["step"] > group["n_warmup"] \
                    else (group["lr"] * state["step"] / group["n_warmup"])

                if rhot > 4:  # variance is tractable - adaptive momentum
                    # bias corrected second moment running average
                    c_vt = vt.div(1 - (beta2 ** state["step"])).pow(0.5)
                    # variance rectification
                    rt = (((rhot - 4) * (rhot - 2) * rho) /
                          ((rho - 4) * (rho - 2) * rhot)) ** 0.5
                    p.data.addcdiv_(-alphat*rt, c_mt, c_vt.add_(group["eps"]))

                else:  # un-adapted momentum
                    p.data.add_(-alphat, c_mt)
        return loss
