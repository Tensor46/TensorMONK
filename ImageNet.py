""" tensorMONK's :: ImageNet                                                 """

from __future__ import print_function, division
import sys
import timeit
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from core import *
import torch.optim as neural_optimizer
# ============================================================================ #


def train():
    args = parse_args()
    tensor_size = (1, 3, 224, 224)
    file_name = "./models/" + args.Architecture.lower()
    embedding_net, embedding_net_kwargs = NeuralArchitectures.Models("resnext50")

    train_loader, n_labels = NeuralEssentials.FolderITTR(args.trainDataPath, args.BSZ, tensor_size, args.cpus,
                                                         functions=[], random_flip=True)
    test_loader, n_labels = NeuralEssentials.FolderITTR(args.testDataPath, args.BSZ, tensor_size, args.cpus,
                                                        functions=[], random_flip=False)

    Model = NeuralEssentials.MakeCNN(file_name, tensor_size, n_labels,
                                     embedding_net=embedding_net,
                                     embedding_net_kwargs=embedding_net_kwargs,
                                     loss_net=NeuralLayers.CategoricalLoss,
                                     loss_net_kwargs={"type": args.loss_type, "distance": args.loss_distance},
                                     default_gpu=args.default_gpu, gpus=args.gpus,
                                     ignore_trained=args.ignore_trained)
    params = list(Model.netEmbedding.parameters()) + list(Model.netLoss.parameters())
    if args.optimizer.lower() == "adam":
        optimizer = neural_optimizer.Adam(params)
    elif args.optimizer.lower() == "sgd":
        optimizer = neural_optimizer.SGD(params, lr=args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        timer = timeit.default_timer()
        Model.netEmbedding.train()
        Model.netLoss.train()
        max_i = 1000
        for i, (tensor, targets) in enumerate(train_loader):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(Variable(tensor))
            loss, (top1, top5) = Model.netLoss((features, Variable(targets)))
            # loss = margin_loss / features.size(0)
            loss.backward()
            optimizer.step()

            # updating all meters
            Model.meterTop1.append(float(top1.cpu().data.numpy() if torch.__version__.startswith("0.4")
                                         else top1.cpu().data.numpy()[0]))
            Model.meterTop5.append(float(top5.cpu().data.numpy() if torch.__version__.startswith("0.4")
                                         else top5.cpu().data.numpy()[0]))
            Model.meterLoss.append(float(loss.cpu().data.numpy() if torch.__version__.startswith("0.4")
                                         else loss.cpu().data.numpy()[0]))

            Model.meterSpeed.append(int(float(args.BSZ)/(timeit.default_timer()-timer)))
            timer = timeit.default_timer()

            print("... {:6d} :: Cost {:2.3f} :: Top1/Top5 - {:3.2f}/{:3.2f} :: {:4d} I/S         ".format(
                  Model.meterIterations, Model.meterLoss[-1], Model.meterTop1[-1],
                  Model.meterTop5[-1], Model.meterSpeed[-1]), end="\r")
            sys.stdout.flush()
            max_i = i

        # save every epoch and print the average of epoch
        print("... {:6d} :: Cost {:1.3f} :: Top1/Top5 - {:3.2f}/{:3.2f} :: {:4d} I/S     ".format(Model.meterIterations,
              np.mean(Model.meterLoss[-max_i:]), np.mean(Model.meterTop1[-max_i:]),
              np.mean(Model.meterTop5[-max_i:]), int(np.mean(Model.meterSpeed[-max_i:]))))
        NeuralEssentials.SaveModel(Model)

        test_top1, test_top5 = [], []
        Model.netEmbedding.eval()
        Model.netLoss.eval()
        for i, (tensor, targets) in enumerate(test_loader):
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(Variable(tensor))
            loss, (top1, top5) = Model.netLoss((features, Variable(targets)))

            test_top1.append(float(top1.cpu().data.numpy() if torch.__version__.startswith("0.4")
                             else top1.cpu().data.numpy()[0]))
            test_top5.append(float(top5.cpu().data.numpy() if torch.__version__.startswith("0.4")
                             else top5.cpu().data.numpy()[0]))
        print("... Test accuracy - {:3.2f}/{:3.2f} ".format(np.mean(test_top1), np.mean(test_top5)))
        Model.netEmbedding.train()
        Model.netLoss.train()

    print("\nDone with training")
    return Model
# ============================================================================ #


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet using TensorMONK!!!")
    parser.add_argument("-A", "--Architecture", type=str, default="residual50",
                        choices=["residual18", "residual34",
                                 "residual50", "residual101", "residual152",
                                 "resnext50", "resnext101", "resnext152",
                                 "seresidual50", "seresidual101", "seresidual152",
                                 "inceptionv4", "mobilev1", "mobilev2",
                                 "shuffle1", "shuffle2", "shuffle3", "shuffle4", "shuffle8"])

    parser.add_argument("-B", "--BSZ", type=int,  default=32)
    parser.add_argument("-E", "--Epochs", type=int,  default=6)

    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--learningRate", type=float, default=0.06)

    parser.add_argument("--loss_type", type=str, default="entr", choices=["entr", "smax", "tentr", "tsmax", "lmcl"])
    parser.add_argument("--loss_distance", type=str, default="dot", choices=["cosine", "dot"])

    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=6)

    parser.add_argument("--trainDataPath", type=str,  default="./data/ImageNet/train")
    parser.add_argument("--testDataPath", type=str,  default="./data/ImageNet/validation")
    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    Model = train()
