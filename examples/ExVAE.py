""" Training a Linear/Convolutional VAE """

from __future__ import print_function, division
import sys
import timeit
import argparse
import torch
import tensormonk


def parse_args():
    parser = argparse.ArgumentParser(description="VAEs using tensorMONK!!!")
    parser.add_argument("-A", "--Architecture", type=str, default="cvae",
                        choices=["cvae", "lvae"])
    parser.add_argument("-P", "--Project", type=str, default="mnist",
                        choices=["mnist", "cifar10"])
    parser.add_argument("-B", "--BSZ", type=int, default=32)
    parser.add_argument("-E", "--Epochs", type=int, default=6)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--learningRate", type=float, default=0.01)
    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


def train():
    args = parse_args()
    # get data
    trData, vaData, teData, n_labels, tensor_size = \
        tensormonk.data.DataSets(args.Project, data_path="../data",
                                 n_samples=args.BSZ)
    args.Architecture = args.Architecture.lower()
    file_name = "./models/" + args.Architecture.lower()
    # visdom plots
    from tensormonk.plots import VisPlots
    visplots = VisPlots(file_name.split("/")[-1].split(".")[0])

    if args.Architecture.lower() == "cvae":
        autoencoder_net = tensormonk.architectures.ConvolutionalVAE
        autoencoder_net_kwargs = {"embedding_layers":
                                  [(3, 32, 2), (3, 64, 2), (3, 128, 2)],
                                  "n_latent": 64,
                                  "decoder_final_activation": "tanh",
                                  "pad": True,
                                  "activation": "relu", "normalization": None}

    elif args.Architecture.lower() == "lvae":
        autoencoder_net = tensormonk.architectures.LinearVAE
        autoencoder_net_kwargs = {"embedding_layers": [1024, 512, 256],
                                  "n_latent": 64,
                                  "decoder_final_activation": "tanh",
                                  "activation": "relu"}

    else:
        raise NotImplementedError

    from tensormonk.essentials import MakeModel, SaveModel
    Model = MakeModel(file_name, tensor_size, n_labels,
                      autoencoder_net, autoencoder_net_kwargs,
                      default_gpu=args.default_gpu, gpus=args.gpus,
                      ignore_trained=args.ignore_trained)

    if args.optimizer.lower() == "adam":
        Optimizer = torch.optim.Adam(Model.netEmbedding.parameters())
    elif args.optimizer.lower() == "sgd":
        Optimizer = torch.optim.SGD(Model.netEmbedding.parameters(),
                                    lr=args.learningRate)
    else:
        raise NotImplementedError

    # Usual training
    for _ in range(args.Epochs):
        Timer = timeit.default_timer()
        Model.netEmbedding.train()
        for i, (tensor, targets) in enumerate(trData):
            Model.meterIterations += 1

            # forward pass and parameter update
            Model.netEmbedding.zero_grad()
            encoded, mu, log_var, latent, decoded, kld, mse = \
                Model.netEmbedding(tensor)
            loss = kld * 0.1 + mse
            loss.backward()
            Optimizer.step()

            if args.Architecture == "cvae":  # l2 weights
                Model.netEmbedding.regularize_weights()

            # updating all meters
            Model.meterLoss.append(float(loss.cpu().data.numpy()))
            kld = float(kld.cpu().data.numpy())
            mse = float(mse.cpu().data.numpy())

            Model.meterSpeed.append(int(float(args.BSZ) /
                                        (timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()

            print("... {:6d} :: Cost {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S    ".
                  format(Model.meterIterations, Model.meterLoss[-1], kld, mse,
                         Model.meterSpeed[-1]), end="\r")
            sys.stdout.flush()

            # weight visualization
            if i % 100 == 0:
                visplots.show_weights(Model.netEmbedding.state_dict(),
                                      png_name=file_name)
                original = tensor[:min(32, tensor.size(0))].data.cpu()
                reconstructed = decoded[:min(32, tensor.size(0))].cpu().data

                if original.dim != 4:
                    original = original.view(original.size(0),
                                             *tensor_size[1:])
                if reconstructed.dim != 4:
                    reconstructed = reconstructed.view(reconstructed.size(0),
                                                       *tensor_size[1:])

                visplots.show_images(torch.cat((original, reconstructed), 0),
                                     vis_name="images",
                                     png_name=file_name + ".png",
                                     normalize=True)

        # save every epoch and print the average of epoch
        print("... {:6d} :: Cost {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S    ".
              format(Model.meterIterations, Model.meterLoss[-1], kld, mse,
                     Model.meterSpeed[-1]))
        SaveModel(Model)
        Timer = timeit.default_timer()
    print("\nDone with training")


if __name__ == '__main__':
    train()
