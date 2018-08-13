""" tensorMONK's :: ExVAE                                                 """

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
import torchvision.utils as show_utils
from core import *
import torch.optim as neuralOptimizer
#==============================================================================#


def trainMONK():
    args = parse_args()
    if args.Project.lower() == "mnist":
        tensor_size = (1, 1, 28, 28)
        trDataLoader, teDataLoader, n_labels = NeuralEssentials.MNIST("../data/MNIST",  tensor_size, args.BSZ, args.cpus)
    elif args.Project.lower() == "cifar10":
        tensor_size = (1, 3, 32, 32)
        trDataLoader, teDataLoader, n_labels = NeuralEssentials.CIFAR10("../data/CIFAR10",  tensor_size, args.BSZ, args.cpus)
    file_name = "./models/" + args.Architecture.lower()

    if args.Architecture.lower() == "cvae":
        autoencoder_net = NeuralArchitectures.ConvolutionalVAE
        autoencoder_net_kwargs = {"embedding_layers" : [(3, 32, 2), (3, 64, 2), (3, 128, 2),], "n_latent" : 64,
                                  "decoder_final_activation" : "tanh", "pad" : True, "activation" : "relu", "normalization" : None}
    elif args.Architecture.lower() == "lvae":
        autoencoder_net = NeuralArchitectures.LinearVAE
        autoencoder_net_kwargs = {"embedding_layers" : [1024, 512,], "n_latent" : 32,
                                  "decoder_final_activation" : "tanh", "activation" : "relu", }
    else:
        raise NotImplementedError

    Model = NeuralEssentials.MakeModel(file_name, tensor_size, n_labels,
                                       autoencoder_net, autoencoder_net_kwargs,
                                       default_gpu=args.default_gpu, gpus=args.gpus,
                                       ignore_trained=args.ignore_trained)

    if args.optimizer.lower() == "adam":
        Optimizer = neuralOptimizer.Adam(Model.netEmbedding.parameters())
    elif args.optimizer.lower() == "sgd":
        Optimizer = neuralOptimizer.SGD(Model.netEmbedding.parameters(), lr= args.learningRate)
    else:
        raise NotImplementedError

    if args.meta_learning:
        transformer = NeuralLayers.ObfuscateDecolor(tensor_size, 0.4, 0.6, 0.5)

    # Usual training
    for _ in range(args.Epochs):
        Timer  = timeit.default_timer()
        Model.netEmbedding.train()
        for i,(tensor, targets) in enumerate(trDataLoader):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            if args.meta_learning:
                org_tensor = Variable(tensor)
                tensor = transformer(org_tensor)
                encoded, mu, log_var, latent, decoded, kld, mse = Model.netEmbedding((org_tensor, tensor))
            else:
                encoded, mu, log_var, latent, decoded, kld, mse = Model.netEmbedding(Variable(tensor))
            loss = kld * 0.1 + mse
            loss.backward()
            Optimizer.step()

            # updating all meters
            Model.meterLoss.append(float(loss.cpu().data.numpy() if torch.__version__.startswith("0.4") else loss.cpu().data.numpy()[0]))
            kld = float(kld.cpu().data.numpy() if torch.__version__.startswith("0.4") else kld.cpu().data.numpy()[0])
            mse = float(mse.cpu().data.numpy() if torch.__version__.startswith("0.4") else mse.cpu().data.numpy()[0])

            Model.meterSpeed.append(int(float(args.BSZ)/(timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()

            print("... {:6d} :: Cost {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S         ".format(Model.meterIterations,
                  Model.meterLoss[-1], kld, mse, Model.meterSpeed[-1]),end="\r")
            sys.stdout.flush()
            if i%100 == 0:
                original = tensor[:min(32,tensor.size(0))].cpu()
                reconstructed = decoded[:min(32,tensor.size(0))].cpu().data

                if original.dim !=4 :
                    original = original.view(original.size(0), *tensor_size[1:])
                if reconstructed.dim !=4 :
                    reconstructed = reconstructed.view(reconstructed.size(0), *tensor_size[1:])

                original = (original - original.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) / \
                           (original.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - original.min(2, keepdim=True)[0].min(3, keepdim=True)[0])
                reconstructed = (reconstructed - reconstructed.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) / \
                                (reconstructed.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - reconstructed.min(2, keepdim=True)[0].min(3, keepdim=True)[0])

                show_utils.save_image(torch.cat([original, reconstructed], 0), "./models/CVAE_train.png", normalize=True)

        # save every epoch and print the average of epoch
        print("... {:6d} :: Cost {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S         ".format(Model.meterIterations,
              Model.meterLoss[-1], kld, mse, Model.meterSpeed[-1]))
        NeuralEssentials.SaveModel(Model)
        Timer = timeit.default_timer()

    print("\nDone with training")
    return Model

# ============================================================================ #
def parse_args():
    parser = argparse.ArgumentParser(description="VAEs using tensorMONK!!!")
    parser.add_argument("-A", "--Architecture", type=str, default="cvae", choices=["cvae", "lvae",])
    parser.add_argument("-P", "--Project", type=str, default="mnist", choices=["mnist", "cifar10",])

    parser.add_argument("-B", "--BSZ", type=int, default=32)
    parser.add_argument("-E", "--Epochs", type=int, default=6)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd",])
    parser.add_argument("--learningRate", type=float, default=0.01)

    parser.add_argument("--meta_learning", action="store_true")

    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=6)

    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    Model = trainMONK()
