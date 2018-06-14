""" tensorMONK's :: SimpleMNIST                                              """

from __future__ import print_function,division
import os
import sys
import timeit
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image as ImPIL
from core import *
import torch.optim as neuralOptimizer
#==============================================================================#


def trainMONK(args):
    tensor_size = (1, 1, 28, 28)
    trDataLoader, teDataLoader, n_labels = NeuralEssentials.MNIST(args.trainDataPath,  tensor_size, args.BSZ, args.cpus)
    file_name = "./models/" + args.Architecture.lower()
    Model = NeuralEssentials.MakeCNN(file_name, tensor_size, n_labels,
                                       embedding_net=NeuralArchitectures.SimpleNet,
                                       embedding_net_kwargs={},
                                       loss_net=NeuralLayers.CategoricalLoss,
                                       loss_net_kwargs={"type" : args.loss_type, "distance" : args.loss_distance},
                                       default_gpu=args.default_gpu, gpus=args.gpus,
                                       ignore_trained=args.ignore_trained)
    params = Model.netEmbedding.parameters() + Model.netLoss.parameters()
    if args.optimizer.lower() == "adam":
        Optimizer = neuralOptimizer.Adam(params)
    elif args.optimizer.lower() == "sgd":
        Optimizer = neuralOptimizer.SGD(params, lr= args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        Timer  = timeit.default_timer()
        Model.netEmbedding.train()
        Model.netLoss.train()
        for i,(tensor, targets) in enumerate(trDataLoader):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding( Variable(tensor) )
            loss, (top1, top5) = Model.netLoss( (features, Variable(targets)) )
            # loss = margin_loss / features.size(0)
            loss.backward()
            Optimizer.step()

            # updating all meters
            Model.meterTop1.append(float(top1.cpu().data.numpy() if torch.__version__.startswith("0.4") else top1.cpu().data.numpy()[0]))
            Model.meterTop5.append(float(top5.cpu().data.numpy() if torch.__version__.startswith("0.4") else top5.cpu().data.numpy()[0]))
            Model.meterLoss.append(float(loss.cpu().data.numpy() if torch.__version__.startswith("0.4") else loss.cpu().data.numpy()[0]))

            Model.meterSpeed.append(int(float(args.BSZ)/(timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()

            print("... {:6d} :: Cost {:2.3f} :: Top1/Top5 - {:3.2f}/{:3.2f} :: {:4d} I/S         ".format(Model.meterIterations,
                Model.meterLoss[-1], Model.meterTop1[-1], Model.meterTop5[-1], Model.meterSpeed[-1]),end="\r")
            sys.stdout.flush()

        # save every epoch and print the average of epoch
        print("... {:6d} :: Cost {:1.3f} :: Top1/Top5 - {:3.2f}/{:3.2f} :: {:4d} I/S     ".format(Model.meterIterations,
                    np.mean(Model.meterLoss[-i:]), np.mean(Model.meterTop1[-i:]),
                    np.mean(Model.meterTop5[-i:]), int(np.mean(Model.meterSpeed[-i:]))))
        NeuralEssentials.SaveModel(Model)

        test_top1, test_top5 = [], []
        Model.netEmbedding.eval()
        Model.netLoss.eval()
        for i,(tensor, targets) in enumerate(teDataLoader):
            Model.meterIterations += 1
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding( Variable(tensor) )
            loss, (top1, top5) = Model.netLoss( (features, Variable(targets)) )

            test_top1.append(float(top1.cpu().data.numpy() if torch.__version__.startswith("0.4") else top1.cpu().data.numpy()[0]))
            test_top5.append(float(top5.cpu().data.numpy() if torch.__version__.startswith("0.4") else top5.cpu().data.numpy()[0]))
        print("... Test accuracy - {:3.2f}/{:3.2f} ".format(np.mean(test_top1), np.mean(test_top5)))
        Model.netEmbedding.train()
        Model.netLoss.train()
        Timer = timeit.default_timer()

    print("\nDone with training")
    return Model

# class argsjump:
#     Architecture = "simplenet"
#     BSZ = 32
#     Epochs = 6
#     optimizer = "sgd"
#     learningRate = .06
#     loss_type = "lmcl"
#     loss_distance = "dot"
#     default_gpu = 1
#     gpus = 1
#     cpus = 6
#     trainDataPath = "./data"
#     testDataPath = "./data"
#     ignore_trained = True
# args = argsjump()

# ============================================================================ #
def parse_args():
    parser = argparse.ArgumentParser(description="SimpleMNIST using TensorMONK!!!")
    parser.add_argument("-A","--Architecture", type=str, default="simplenet")

    parser.add_argument("-B","--BSZ", type=int,  default=32)
    parser.add_argument("-E","--Epochs", type=int,  default=6)

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd",])
    parser.add_argument("--learningRate", type=float, default=0.06)

    parser.add_argument("--loss_type", type=str, default="entr", choices=["entr", "smax", "tentr", "tsmax", "lmcl"])
    parser.add_argument("--loss_distance", type=str, default="dot", choices=["cosine", "dot"])

    parser.add_argument("--default_gpu", type=int,  default=1)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=6)

    parser.add_argument("--trainDataPath", type=str,  default="./data")
    parser.add_argument("--testDataPath", type=str,  default="./data")
    parser.add_argument("-I","--ignore_trained", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Model = trainMONK(args)
