""" tensorMONK's :: mellowID                                                 """

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


def trainMONK(args):
    tensor_size = (1, 1, 28, 28)
    trDataLoader, teDataLoader, n_labels = NeuralEssentials.MNIST(args.trainDataPath,  tensor_size, args.BSZ, args.cpus)
    file_name = "./models/" + args.Architecture.lower()

    autoencoder_net = NeuralArchitectures.ConvolutionalVAE
    autoencoder_net_kwargs = {"embedding_layers" : [(3, 16, 2), (3, 32, 2),], "n_latent" : 32,
                              "decoder_final_activation" : "tanh", "pad" : True, "activation" : "relu",
                              "batch_nm" : False}

    Model = NeuralEssentials.MakeAE(file_name, tensor_size, n_labels,
                                    autoencoder_net, autoencoder_net_kwargs,
                                    default_gpu=args.default_gpu, gpus=args.gpus,
                                    ignore_trained=args.ignore_trained)

    if args.optimizer.lower() == "adam":
        Optimizer = neuralOptimizer.Adam(Model.netAE.parameters())
    elif args.optimizer.lower() == "sgd":
        Optimizer = neuralOptimizer.SGD(Model.netAE.parameters(), lr= args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        Timer  = timeit.default_timer()
        Model.netAE.train()
        for i,(tensor, targets) in enumerate(trDataLoader):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netAE.zero_grad()
            encoded, mu, log_var, latent, decoded, kld, mse = Model.netAE(Variable(tensor))
            loss = kld * 0.005 + mse
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
                original = tensor[:min(16,tensor.size(0))].cpu()
                original = (original - original.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) / \
                           (original.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - original.min(2, keepdim=True)[0].min(3, keepdim=True)[0])
                reconstructed = decoded[:min(16,tensor.size(0))].cpu().data
                reconstructed = (reconstructed - reconstructed.min(2, keepdim=True)[0].min(3, keepdim=True)[0]) / \
                                (reconstructed.max(2, keepdim=True)[0].max(3, keepdim=True)[0] - reconstructed.min(2, keepdim=True)[0].min(3, keepdim=True)[0])
                show_utils.save_image(torch.cat([original, reconstructed], 0), "./models/CVAE_train.png", normalize=True)


        # save every epoch and print the average of epoch
        print("... {:6d} :: Cost {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S         ".format(Model.meterIterations,
              Model.meterLoss[-1], kld, mse, Model.meterSpeed[-1]))
        NeuralEssentials.SaveModel(Model)

        # test_top1, test_top5 = [], []
        # Model.netAE.eval()
        # for i,(tensor, targets) in enumerate(teDataLoader):
        #
        #     Model.netEmbedding.zero_grad()
        #     Model.netLoss.zero_grad()
        #     encoded, mu, log_var, latent, decoded, kld, mse = Model.netAE(Variable(tensor))
        #
        #
        #
        #     test_top1.append(float(top1.cpu().data.numpy() if torch.__version__.startswith("0.4") else top1.cpu().data.numpy()[0]))
        #     test_top5.append(float(top5.cpu().data.numpy() if torch.__version__.startswith("0.4") else top5.cpu().data.numpy()[0]))
        # print("... Test accuracy - {:3.2f}/{:3.2f} ".format(np.mean(test_top1), np.mean(test_top5)))
        # Model.netEmbedding.train()
        # Model.netLoss.train()
        Timer = timeit.default_timer()

    print("\nDone with training")
    return Model

# ============================================================================ #
def parse_args():
    parser = argparse.ArgumentParser(description="mellowID using tensorMONK!!!")
    parser.add_argument("-A", "--Architecture", type=str, default="cvae")
    parser.add_argument("-B", "--BSZ", type=int, default=32)
    parser.add_argument("-E", "--Epochs", type=int, default=6)

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd",])
    parser.add_argument("--learningRate", type=float, default=0.06)

    parser.add_argument("--default_gpu", type=int,  default=1)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=6)

    parser.add_argument("--trainDataPath", type=str, default="./data/MNIST")
    parser.add_argument("--testDataPath", type=str, default="./data/MNIST")

    parser.add_argument("--replicate_paper", action="store_true")

    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Model = trainMONK(args)
