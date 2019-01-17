""" Training Progressing growth of GANs on CIFAR10 """

from __future__ import print_function, division
import os
import sys
import math
import timeit
import argparse
import numpy as np
import tensormonk
import torch.nn.functional as F

MSG = "... {:6d} :: Cost df/dr/g {:2.3f}/{:2.3f}/{:2.3f} :: {:4d} I/S      "


def parse_args():
    parser = argparse.ArgumentParser(description="PGGAN using TensorMONK!!!")
    parser.add_argument("-B", "--BSZ", type=int,  default=64)
    parser.add_argument("--learningRate", type=float, default=0.0001)

    parser.add_argument("--default_gpu", type=int,  default=1)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=4)

    parser.add_argument("--levels", type=int,  default=4)
    parser.add_argument("--growth_rate", type=int,  default=64)
    parser.add_argument("--l1_iterations", type=int,  default=100000)
    parser.add_argument("-I", "--ignore_trained", action="store_true")
    return parser.parse_args()


def train():
    def cost_sampler(i):
        return min(max(1, math.ceil(i / 1000.)), 1000)

    sampler = 1

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
    embedding_net_kwargs = {"levels": args.levels, "l1_size": l1_size,
                            "l1_iterations": args.l1_iterations,
                            "growth_rate": args.growth_rate}
    from tensormonk.essentials import MakeModel, SaveModel
    Model = MakeModel(file_name, tensor_size, n_labels,
                      embedding_net=tensormonk.architectures.PGGAN,
                      embedding_net_kwargs=embedding_net_kwargs,
                      default_gpu=args.default_gpu,
                      gpus=args.gpus,
                      ignore_trained=args.ignore_trained)

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
            Model.netEmbedding.NET46.iterations.add_(1)
            Model.netEmbedding.NET46.update()

            # forward pass and parameter update
            latent = Model.netEmbedding.NET46.noisy_latent(args.BSZ)
            fake = Model.netEmbedding(latent)

            # A constrain on the updates with respect to loss, so, nither wins!
            # train discriminator with real sample
            d_loss_real = Model.netEmbedding(tensor).mul(-1).add(1).mean()
            Model.netEmbedding.zero_grad()
            if d_loss_real.data.cpu().numpy() >= .2:
                d_loss_real.backward()
                Model.netEmbedding.NET46.d_optimizer.step()

            # train discriminator with fake sample
            d_loss_fake = Model.netEmbedding(fake.detach()).mean()
            Model.netEmbedding.zero_grad()
            if d_loss_fake.data.cpu().numpy() >= .2:
                d_loss_fake.backward()
                Model.netEmbedding.NET46.d_optimizer.step()

            # generate a fake sample
            g_loss = Model.netEmbedding(fake).mul(-1).add(1).mean()
            Model.netEmbedding.zero_grad()
            if g_loss.data.cpu().numpy() >= .1:
                g_loss.backward()
                Model.netEmbedding.NET46.g_optimizer.step()

            # Visdom visualization + gifs
            if Model.meterIterations % 20 == 0:
                level_id = str(Model.netEmbedding.NET46.current_level)
                if png_count == 20:
                    list_images = [os.path.join("./models", x) for x in
                                   next(os.walk("./models"))[2] if level_id
                                   in x and x.endswith(".png") and
                                   file_name.split("/")[-1] in x]
                    make_gif(list_images, file_name + "-" + level_id + ".gif")
                    png_count = 0
                name = "level" + level_id
                fake = F.interpolate(fake if fake.size(0) < 64 else fake[:64],
                                     size=tensor_size[2:])
                visplots.show_images(fake.data.cpu(), name, file_name + "-" +
                                     name + "_" + str(png_count)+".png")
                png_count += 1

            # updating all meters
            Model.meterLoss.append(float(g_loss.cpu().data.numpy()))
            Model.meterLoss.append(float(d_loss_real.cpu().data.numpy()))
            Model.meterLoss.append(float(d_loss_fake.cpu().data.numpy()))
            Model.meterSpeed.append(int(float(args.BSZ) /
                                        (timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()
            print(MSG.format(Model.meterIterations,
                             Model.meterLoss[-1],
                             Model.meterLoss[-2],
                             Model.meterLoss[-3],
                             Model.meterSpeed[-1]), end="\r")
            sys.stdout.flush()

            # save and track
            if Model.meterIterations % 2000 == 0:
                if not np.sum(np.isnan(Model.meterLoss[-50:])):
                    SaveModel(Model)
                print(MSG.format(Model.meterIterations,
                                 np.mean(Model.meterLoss[2::3][-1000:]),
                                 np.mean(Model.meterLoss[1::3][-1000:]),
                                 np.mean(Model.meterLoss[0::3][-1000:]),
                                 int(np.mean(Model.meterSpeed[-2000:]))))
                print(" ... level= {:d}, transition= {}, alpha= {:1.4}".format(
                    Model.netEmbedding.NET46.current_level, "ON" if
                    Model.netEmbedding.NET46.transition else "OFF",
                    Model.netEmbedding.NET46.alpha))

            if Model.meterIterations % sampler == 0:
                # Cost plot
                sampler = cost_sampler(Model.meterIterations)
                cost_samples = Model.meterLoss[0::3][0::sampler]
                xaxis = np.arange(0, len(cost_samples)*sampler, sampler)
                visplots.visplots.line(Y=np.array(cost_samples), X=xaxis,
                                       opts={"title": "gCost"}, update=None,
                                       win="gCost")
                cost_samples = Model.meterLoss[1::3][0::sampler]
                visplots.visplots.line(Y=np.array(cost_samples), X=xaxis,
                                       opts={"title": "dCost-real"},
                                       update=None, win="dCost-real")
                cost_samples = Model.meterLoss[2::3][0::sampler]
                visplots.visplots.line(Y=np.array(cost_samples), X=xaxis,
                                       opts={"title": "dCost-fake"},
                                       update=None, win="dCost-fake")
    print("\nDone with training")


if __name__ == '__main__':
    train()
