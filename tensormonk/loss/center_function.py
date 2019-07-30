
import torch
from torch.autograd.function import Function


class CenterFunction(Function):

    @staticmethod
    def forward(ctx, tensor, targets, centers, alpha, scale):
        ctx.save_for_backward(tensor, targets, centers, alpha, scale)
        target_centers = centers.index_select(0, targets)
        return scale / 2 * (tensor - target_centers).pow(2).sum(1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        tensor, targets, centers, alpha, scale = ctx.saved_variables
        targets = targets.long()
        n = targets.size(0)
        grad_centers = torch.zeros(centers.size()).to(tensor.device)
        delta = centers.index_select(0, targets) - tensor
        counter = torch.histc(targets.float(), bins=centers.shape[1], min=0,
                              max=centers.shape[1]-1)
        grad_centers.scatter_add_(0, targets.view(-1, 1).expand(tensor.size()),
                                  delta)
        idx = counter.nonzero().view(-1)
        grad_centers[idx] = grad_centers[idx] / counter[idx].unsqueeze(1)
        grad_centers.mul_(alpha)
        grad_tensor = - grad_output * delta / n
        return grad_tensor, None, grad_centers, None, None
