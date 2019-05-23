""" Training architectures on 2012 ImageNet """

from __future__ import print_function, division
import sys
import timeit
import argparse
import numpy as np
import torch
import tensormonk


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet using TensorMONK!")
    parser.add_argument("-A", "--Architecture", type=str, default="residual50",
                        choices=["residual18", "residual34",
                                 "residual50", "residual101", "residual152",
                                 "resnext50", "resnext101", "resnext152",
                                 "seresidual50", "seresidual101",
                                 "seresidual152",
                                 "inceptionv4", "mobilev1", "mobilev2",
                                 "shuffle1", "shuffle2", "shuffle3",
                                 "shuffle4", "shuffle8"])
    parser.add_argument("-B", "--BSZ", type=int,  default=32)
    parser.add_argument("-E", "--Epochs", type=int,  default=6)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd"])
    parser.add_argument("--learningRate", type=float, default=0.06)

    parser.add_argument("--loss_type", type=str, default="entr",
                        choices=["entr", "smax", "tentr", "tsmax", "lmcl"])
    parser.add_argument("--loss_distance", type=str, default="dot",
                        choices=["cosine", "dot"])
    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=6)

    parser.add_argument("--trainDataPath", type=str,
                        default="./data/ImageNet/train")
    parser.add_argument("--testDataPath", type=str,
                        default="./data/ImageNet/validation")
    parser.add_argument("-I", "--ignore_trained", action="store_true")
    return parser.parse_args()


def train():
    args = parse_args()
    tensor_size = (1, 3, 224, 224)

    file_name = "./models/" + args.Architecture.lower()
    embedding_net, embedding_net_kwargs = \
        tensormonk.architectures.Models(args.Architecture.lower())

    train_loader, n_labels = \
        tensormonk.data.FolderITTR(args.trainDataPath, args.BSZ, tensor_size,
                                   args.cpus, functions=[], random_flip=True)
    test_loader, n_labels = \
        tensormonk.data.FolderITTR(args.testDataPath, args.BSZ, tensor_size,
                                   args.cpus, functions=[], random_flip=False)

    from tensormonk.essentials import MakeModel, SaveModel
    Model = MakeModel(file_name, tensor_size, n_labels,
                      embedding_net=embedding_net,
                      embedding_net_kwargs=embedding_net_kwargs,
                      loss_net=tensormonk.loss.CategoricalLoss,
                      loss_net_kwargs={"type": args.loss_type,
                                       "distance": args.loss_distance},
                      default_gpu=args.default_gpu,
                      gpus=args.gpus,
                      ignore_trained=args.ignore_trained)

    params = list(Model.netEmbedding.parameters()) + \
        list(Model.netLoss.parameters())
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        timer = timeit.default_timer()
        Model.netEmbedding.train()
        Model.netLoss.train()
        for i, (tensor, targets) in enumerate(train_loader):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(tensor)
            loss, (top1, top5) = Model.netLoss((features, targets))
            loss.backward()
            optimizer.step()

            # updating all meters
            Model.meterTop1.append(float(top1.cpu().data.numpy()))
            Model.meterTop5.append(float(top5.cpu().data.numpy()))
            Model.meterLoss.append(float(loss.cpu().data.numpy()))

            Model.meterSpeed.append(int(float(args.BSZ) /
                                    (timeit.default_timer()-timer)))
            timer = timeit.default_timer()

            print("... {:6d} :: ".format(Model.meterIterations) +
                  "Cost {:2.3f} :: ".format(Model.meterLoss[-1]) +
                  "Top1/Top5 - {:3.2f}/{:3.2f}".format(Model.meterTop1[-1],
                                                       Model.meterTop5[-1],) +
                  " :: {:4d} I/S    ".format(Model.meterSpeed[-1]), end="\r")
            sys.stdout.flush()

        # save every epoch and print the average of epoch
        mean_loss = np.mean(Model.meterLoss[-i:])
        mean_top1 = np.mean(Model.meterTop1[-i:])
        mean_top5 = np.mean(Model.meterTop5[-i:])
        mean_speed = int(np.mean(Model.meterSpeed[-i:]))
        print("... {:6d} :: ".format(Model.meterIterations) +
              "Cost {:2.3f} :: ".format(mean_loss) +
              "Top1/Top5 - {:3.2f}/{:3.2f}".format(mean_top1, mean_top5) +
              " :: {:4d} I/S    ".format(mean_speed))
        # save model
        SaveModel(Model)

        test_top1, test_top5 = [], []
        Model.netEmbedding.eval()
        Model.netLoss.eval()
        for i, (tensor, targets) in enumerate(test_loader):
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(tensor)
            loss, (top1, top5) = Model.netLoss((features, targets))
            test_top1.append(float(top1.cpu().data.numpy()))
            test_top5.append(float(top5.cpu().data.numpy()))
        print("... Test accuracy - {:3.2f}/{:3.2f} ".format(
            np.mean(test_top1), np.mean(test_top5)))
        Model.netEmbedding.train()
        Model.netLoss.train()

    print("\nDone with training")


if __name__ == '__main__':
    train()
