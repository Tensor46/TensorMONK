""" Training a 3 layer cnn on mnist and fashion mnist """

from __future__ import print_function, division
import sys
import timeit
import argparse
import numpy as np
import torch
import tensormonk


def parse_args():
    parser = argparse.ArgumentParser(description="SimpleMNIST")
    parser.add_argument("-D", "--dataset", type=str, default="fashionmnist",
                        choices=["mnist", "fashionmnist"])

    parser.add_argument("-B", "--BSZ", type=int, default=32)
    parser.add_argument("-E", "--Epochs", type=int, default=6)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd"])
    parser.add_argument("--learningRate", type=float, default=0.06)

    parser.add_argument("--loss_type", type=str, default="entr",
                        choices=["entr", "smax", "tentr", "tsmax", "lmcl"])
    parser.add_argument("--loss_measure", type=str, default="dot",
                        choices=["cosine", "dot"])

    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=6)

    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


def train():
    r"""An example to train 3 layer cnn on mnist and fashion mnist.
    """
    args = parse_args()
    from tensormonk.data import DataSets
    trData, vaData, teData, n_labels, tensor_size = \
        DataSets(args.dataset, data_path="../data", n_samples=args.BSZ)

    file_name = "./models/simplenet"
    from tensormonk.plots import VisPlots
    visplots = VisPlots(file_name.split("/")[-1].split(".")[0])

    from tensormonk.essentials import MakeModel, SaveModel
    Model = MakeModel(file_name,
                      tensor_size,
                      n_labels,
                      embedding_net=tensormonk.architectures.SimpleNet,
                      embedding_net_kwargs={},
                      loss_net=tensormonk.loss.Categorical,
                      loss_net_kwargs={"type": args.loss_type,
                                       "measure": args.loss_measure},
                      default_gpu=args.default_gpu,
                      gpus=args.gpus,
                      ignore_trained=args.ignore_trained)

    params = list(Model.netEmbedding.parameters()) + \
        list(Model.netLoss.parameters())

    if args.optimizer.lower() == "adam":
        Optimizer = torch.optim.Adam(params, amsgrad=True)
    elif args.optimizer.lower() == "sgd":
        Optimizer = torch.optim.SGD(params, lr=args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        Timer = timeit.default_timer()

        Model.netEmbedding.train()
        Model.netLoss.train()

        for i, (tensor, targets) in enumerate(trData):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(tensor)
            loss, (top1, top5) = Model.netLoss((features, targets))
            loss.backward()
            Optimizer.step()

            # updating all meters
            Model.meterTop1.append(float(top1.cpu().data.numpy()))
            Model.meterTop5.append(float(top5.cpu().data.numpy()))
            Model.meterLoss.append(float(loss.cpu().data.numpy()))
            Model.meterSpeed.append(int(float(args.BSZ) /
                                        (timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()

            # weight visualization
            if i % 50 == 0:
                visplots.show_weights(Model.netEmbedding.state_dict(),
                                      png_name=file_name)

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
        for i, (tensor, targets) in enumerate(teData):
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            features = Model.netEmbedding(tensor)
            loss, (top1, top5) = Model.netLoss((features, targets))

            test_top1.append(float(top1.cpu().data.numpy()))
            test_top5.append(float(top5.cpu().data.numpy()))
        print("... Test accuracy - {:3.2f}/{:3.2f}".format(np.mean(test_top1),
                                                           np.mean(test_top5)))
        Model.netEmbedding.train()
        Model.netLoss.train()
        Timer = timeit.default_timer()

    print("\nDone with training")


if __name__ == '__main__':

    train()
