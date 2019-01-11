""" TensorMONK's :: Progressing growth of GANs on CIFAR10                   """

from __future__ import print_function, division
import os
import sys
import timeit
import argparse
import numpy as np
import torch
import tensormonk


def parse_args():
    parser = argparse.ArgumentParser(description="PGGAN using TensorMONK!!!")
    parser.add_argument("-B", "--BSZ", type=int,  default=64)
    parser.add_argument("--learningRate", type=float, default=0.0001)

    parser.add_argument("--default_gpu", type=int,  default=1)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=4)

    parser.add_argument("--levels", type=int,  default=4)
    parser.add_argument("--growth_rate", type=int,  default=32)
    parser.add_argument("--n_embedding", type=int,  default=256)
    parser.add_argument("--l1_iterations", type=int,  default=100000)
    parser.add_argument("-I", "--ignore_trained", action="store_true")
    return parser.parse_args()


def noisy_latent(n, n_embedding):  # noise
    return torch.randn(n, n_embedding)


def train():
    args = parse_args()
    from tensormonk.data import DataSets
    trData, vaData, teData, n_labels, tensor_size = \
        DataSets("cifar10", data_path="../data", n_samples=args.BSZ,
                 normalize=False)
    l1_size = (4, 4)

    # build model and set pggan updater
    file_name = "./models/pggan-cifar10"
    from tensormonk.plots import VisPlots, make_gif
    visplots = VisPlots(file_name.split("/")[-1].split(".")[0])
    embedding_net_kwargs = {"n_embedding": args.n_embedding,
                            "levels": args.levels, "l1_size": l1_size,
                            "l1_iterations": args.l1_iterations,
                            "growth_rate": args.growth_rate}
    from tensormonk.essentials import MakeModel, SaveModel
    Model = MakeModel(file_name, tensor_size, n_labels,
                      embedding_net=tensormonk.architectures.PGGAN,
                      embedding_net_kwargs=embedding_net_kwargs,
                      default_gpu=args.default_gpu,
                      gpus=args.gpus,
                      ignore_trained=args.ignore_trained)

    # optimizers
    g_params, d_params = Model.netEmbedding.NET46.trainable_parameters()
    g_optimizer = torch.optim.Adam(g_params, lr=args.learningRate,
                                   amsgrad=True)
    d_optimizer = torch.optim.Adam(d_params, lr=args.learningRate,
                                   amsgrad=True)

    print(" ... level = {:d},".format(Model.netEmbedding.NET46.current_level) +
          " transition = {},".format("ON" if
                                     Model.netEmbedding.NET46.transition
                                     else "OFF") +
          " alpha = {:1.4}".format(Model.netEmbedding.NET46.alpha))
    print(" total iterations - ", Model.netEmbedding.NET46.total_ittr)
    print("")

    png_count = 0
    # Usual training
    while True:
        Timer = timeit.default_timer()
        Model.netEmbedding.train()
        if Model.meterIterations >= Model.netEmbedding.NET46.total_ittr:
            print(" ... done with training!")
            break
        for i, (tensor, targets) in enumerate(trData):
            if Model.meterIterations >= Model.netEmbedding.NET46.total_ittr:
                break
            Model.meterIterations += 1

            # forward pass and parameter update
            latent = noisy_latent(args.BSZ, args.n_embedding)
            fake = Model.netEmbedding(latent)

            # A constrain on the updates with respect to loss, so, nither wins!
            # train discriminator with real sample
            d_loss_real = Model.netEmbedding(tensor).mul(-1).add(1).mean()
            d_optimizer.zero_grad()
            if d_loss_real.data.cpu().numpy() > .1:
                d_loss_real.backward()
                d_optimizer.step()

            # train discriminator with fake sample
            d_loss_fake = Model.netEmbedding(fake.detach()).mean()
            d_optimizer.zero_grad()
            if d_loss_fake.data.cpu().numpy() > .1:
                d_loss_fake.backward()
                d_optimizer.step()
            d_loss = (d_loss_fake + d_loss_real) / 2

            # generate a fake sample
            g_loss = Model.netEmbedding(fake).mul(-1).add(1).mean()
            g_optimizer.zero_grad()
            if g_loss.data.cpu().numpy() > .1:
                g_loss.backward()
                g_optimizer.step()

            # Visdom visualization + gifs
            if Model.meterIterations % 20 == 0:
                level_id = str(Model.netEmbedding.NET46.current_level)
                if png_count == 20:
                    list_images = [os.path.join("./models", x) for x in
                                   next(os.walk("./models"))[2] if level_id
                                   in x and x.endswith(".png")]
                    make_gif(list_images, file_name + "-" + level_id + ".gif")
                    png_count = 0
                visplots.show_weights(Model.netEmbedding.state_dict())
                name = "level" + level_id
                visplots.show_images(fake.data.cpu(), name, file_name + "-" +
                                     name + "_" + str(png_count)+".png")
                png_count += 1
                visplots.show_images(tensor.data.cpu(), "real")
            Model.netEmbedding.regularize_weights()

            # updating all meters
            Model.meterLoss.append(float(g_loss.cpu().data.numpy()))
            Model.meterLoss.append(float(d_loss.cpu().data.numpy()))
            Model.meterSpeed.append(int(float(args.BSZ) /
                                        (timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()
            print("... {:6d} :: Cost d/g {:2.3f}/{:2.3f} :: {:4d} I/S ".format(
                Model.meterIterations, Model.meterLoss[-1],
                Model.meterLoss[-2], Model.meterSpeed[-1]), end="\r")
            sys.stdout.flush()

            # save and track
            if Model.meterIterations % 2000 == 0:
                if np.mean(Model.meterLoss[1::2][-50:]) > 0.1 and \
                   np.mean(Model.meterLoss[1::2][-50:]) < 0.9 and \
                   np.mean(Model.meterLoss[0::2][-50:]) > 0.1 and \
                   np.mean(Model.meterLoss[0::2][-50:]) < 0.9:
                    # only save when last few g_loss and d_loss are in sensible
                    SaveModel(Model)
                print("... {:6d} :: Cost d/g {:2.3f}/{:2.3f} :: ".format(
                    Model.meterIterations,
                    np.mean(Model.meterLoss[1::2][-1000:]),
                    np.mean(Model.meterLoss[0::2][-1000:])) +
                    "{:4d} I/S  ".format(
                    int(np.mean(Model.meterSpeed[-2000:]))))
                print(" ... level= {:d}, transition= {}, alpha= {:1.4}".format(
                    Model.netEmbedding.NET46.current_level, "ON" if
                    Model.netEmbedding.NET46.transition else "OFF",
                    Model.netEmbedding.NET46.alpha))
    print("\nDone with training")


if __name__ == '__main__':
    train()
