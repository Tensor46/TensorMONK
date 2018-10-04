""" TensorMONK's :: NeuralArchitectures                                      """

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# ============================================================================ #


class NeuralTree(nn.Module):
    """
        A neural tree!
        --------------

        Parameters
        ----------
        tensor_size = 2D or 4D
                      2D - (None/any integer >0, features)
                      4D - (None/any integer >0, channels, height, width)
        n_labels = number of labels or classes
        depth = depth of trees
                Ex: linear layer indices for a tree of depth = 2
                    0
                  1   2
                 3 4 5 6
                a linear requries 7 output neurons (2**(depth+1) - 1)
        dropout = ususal dropout
        network = any custom torch module can be used to produce leaf outputs
                  ( must have output neurons of length 2**(depth+1)-1 )
                  when None, linear + relu + dropout + linear + sigm
    """
    def __init__(self, tensor_size, n_labels, depth, dropout=0.5, network=None):
        super(NeuralTree, self).__init__()

        assert depth > 0, \
            "NeuralTree :: depth must be > 0, given {}".format(depth)

        self.tensor_size = tensor_size
        self.n_labels = n_labels
        self.depth = depth
        self.n_leafs = 2**(depth+1)
        # dividing the linear output to decisions at different levels
        self.decision_per_depth = [2**x for x in range(depth+1)]
        # their indices
        self.indices_per_depth = [list(range(y-x, max(1, y))) for x, y in \
            zip(self.decision_per_depth, np.cumsum(self.decision_per_depth))]

        from core.NeuralLayers import Linear
        # an example - can be any number of layers
        self.tree = nn.Sequential(Linear(tensor_size, (self.n_leafs+1)*4, "relu"),
            Linear((None, (self.n_leafs+1)*4), self.n_leafs - 1, "sigm", dropout)) \
            if network is None else network

        self.weight = nn.Parameter(torch.Tensor(self.n_leafs, n_labels))
        nn.init.xavier_normal_(self.weight)
        self.tensor_size = (None, n_labels)

    def forward(self, tensor):
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)
        BSZ = tensor.size(0)
        # get all leaf responses -- a simple linear layer
        leaf_responses = self.tree(tensor)
        # compute decisions from the final depth
        decision = leaf_responses[:, self.indices_per_depth[0]]
        for x in self.indices_per_depth[1:]:
            decision = decision.unsqueeze(2)
            # true and false of last depth
            decision = torch.cat((decision, 1 - decision), 2).view(BSZ, -1)
            # current depth decisions
            decision = decision.mul(leaf_responses[:, x])
        decision = decision.unsqueeze(2)
        decision = torch.cat((decision, 1 - decision), 2).view(BSZ, -1)
        # predictions
        predictions = decision.unsqueeze(2).mul(F.softmax(self.weight, 1).unsqueeze(0))
        return decision, predictions.sum(1)

# test = NeuralTree((1, 64), 12, 4)
# decision, predictions = test(torch.rand(15, 64))
# predictions.shape
# ============================================================================ #


class NeuralDecisionForest(nn.Module):
    """
        Neural Decision Forest
        ----------------------

        Parameters
        ----------
        tensor_size = check NeuralTree
        n_labels = check NeuralTree
        n_trees = number of trees
        depth = check NeuralTree
        network = check NeuralTree

        A version of https://ieeexplore.ieee.org/document/7410529
    """
    def __init__(self, tensor_size, n_labels, n_trees, depth,
                 dropout=0.5, network=None):
        super(NeuralDecisionForest, self).__init__()

        assert depth > 0, \
            "NeuralDecisionForest :: depth must be > 0, given {}".format(depth)

        self.trees = nn.ModuleList([NeuralTree(tensor_size, n_labels, depth,
            dropout, network) for i in range(n_trees)])
        self.tensor_size = self.trees[0].tensor_size

    def forward(self, tensor):
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)
        predictions = torch.cat([tree(tensor)[1].unsqueeze(2) for tree in self.trees], 2)
        return predictions.mean(2).log()


# test = NeuralDecisionForest((1, 64), 12, 10, 4)
# predictions = test(torch.rand(15, 64))
# predictions.shape
