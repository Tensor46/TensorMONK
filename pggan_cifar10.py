""" TensorMONK's :: Progressing growth of GANs on CIFAR10                    """

from __future__ import print_function,division
import os
import sys
import timeit
import argparse
import numpy as np
import core
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# ============================================================================ #


def train():
    args = parse_args()
    # noise
    noisy_latent = lambda : torch.randn(args.BSZ, args.n_embedding)
    # few basics and get data loaders
    tensor_size = (1, 3, 32, 32)
    l1_size = (4, 4)
    trDataLoader, teDataLoader, n_labels = core.NeuralEssentials.CIFAR10(args.datapath,
                                                tensor_size, args.BSZ, args.cpus, normalize_01=True)

    # build model and set pggan updater
    file_name = "./models/pggan-cifar10"
    visplots = core.NeuralEssentials.VisPlots(file_name)
    Model = core.NeuralEssentials.MakeModel(file_name, tensor_size, n_labels,
                                            embedding_net=core.NeuralArchitectures.PGGAN,
                                            embedding_net_kwargs={"n_embedding": args.n_embedding,
                                                                  "levels": args.levels,
                                                                  "l1_size": l1_size,
                                                                  "l1_iterations": args.l1_iterations,
                                                                  "growth_rate": args.growth_rate},
                                            default_gpu=args.default_gpu, gpus=args.gpus,
                                            ignore_trained=args.ignore_trained)

    # optimizers
    g_optimizer = torch.optim.Adam(Model.netEmbedding.NET46.g_modules.parameters(),
                                   lr=args.learningRate, weight_decay=0.00005, amsgrad=True)
    d_optimizer = torch.optim.Adam(Model.netEmbedding.NET46.d_modules.parameters(),
                                   lr=args.learningRate, weight_decay=0.00005, amsgrad=True)


    print(" ... level = {:d}, transition = {}, alpha = {:1.4}".format(
        Model.netEmbedding.NET46.current_level, "ON" if
        Model.netEmbedding.NET46.transition else "OFF",
        Model.netEmbedding.NET46.alpha))
    print(" total iterations - ", Model.netEmbedding.NET46.max_iterations)
    print(" final output size - ", Model.netEmbedding.NET46.max_tensor_size[2:])
    print("")

    png_count = 0
    # Usual training
    while True:
        Timer  = timeit.default_timer()
        Model.netEmbedding.train()
        if Model.meterIterations >= Model.netEmbedding.NET46.max_iterations:
            print(" ... done with training!")
            break
        for i, (tensor, targets) in enumerate(trDataLoader):
            if Model.meterIterations >= Model.netEmbedding.NET46.max_iterations:
                break
            Model.meterIterations += 1
            Model.netEmbedding.NET46.updates(Model.meterIterations)
            # forward pass and parameter update
            latent = noisy_latent()
            fake = Model.netEmbedding(latent)

            # A constrain on the updates with respect to loss, so, either wins!
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
                if png_count == 20:
                    list_images = [os.path.join("./models", x) for x in
                        next(os.walk("./models"))[2] if level_id in x and x.endswith(".png")]
                    core.NeuralEssentials.MakeGIF(list_images, file_name+"-"+level_id+".gif")
                    png_count = 0
                visplots.show_weights(Model.netEmbedding.state_dict())
                name = "level" + str(Model.netEmbedding.NET46.current_level)
                visplots.show_images(fake.data.cpu(), name,
                    file_name+"-"+name+"_"+str(png_count)+".png")
                png_count += 1
                visplots.show_images(tensor.data.cpu(), "real")

            # updating all meters
            Model.meterLoss.append(float(g_loss.cpu().data.numpy()))
            Model.meterLoss.append(float(d_loss.cpu().data.numpy()))
            Model.meterSpeed.append(int(float(args.BSZ)/(timeit.default_timer()-Timer)))
            Timer = timeit.default_timer()
            print("... {:6d} :: Cost d/g {:2.3f}/{:2.3f} :: {:4d} I/S      ".format(Model.meterIterations,
                Model.meterLoss[-1], Model.meterLoss[-2], Model.meterSpeed[-1]),end="\r")
            sys.stdout.flush()

            # save and track
            if Model.meterIterations % 2000 == 0:
                if np.mean(Model.meterLoss[1::2][-50:]) > 0.1 and \
                   np.mean(Model.meterLoss[1::2][-50:]) < 0.9 and \
                   np.mean(Model.meterLoss[0::2][-50:]) > 0.1 and \
                   np.mean(Model.meterLoss[0::2][-50:]) < 0.9:
                    # only save when last few g_loss and d_loss are in sensible
                    core.NeuralEssentials.SaveModel(Model)
                print("... {:6d} :: Cost d/g {:2.3f}/{:2.3f} :: {:4d} I/S      ".format(Model.meterIterations,
                    np.mean(Model.meterLoss[1::2][-1000:]), np.mean(Model.meterLoss[0::2][-1000:]),
                    int(np.mean(Model.meterSpeed[-2000:]))))
                print(" ... level = {:d}, transition = {}, alpha = {:1.4}".format(
                    Model.netEmbedding.NET46.current_level, "ON" if
                    Model.netEmbedding.NET46.transition else "OFF",
                    Model.netEmbedding.NET46.alpha))
    print("\nDone with training")
# ============================================================================ #


def parse_args():
    parser = argparse.ArgumentParser(description="PGGAN using TensorMONK!!!")

    parser.add_argument("-B","--BSZ", type=int,  default=64)
    parser.add_argument("--learningRate", type=float, default=0.0001)

    parser.add_argument("--default_gpu", type=int,  default=1)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("--cpus", type=int,  default=4)

    parser.add_argument("--levels", type=int,  default=4)
    parser.add_argument("--growth_rate", type=int,  default=32)
    parser.add_argument("--n_embedding", type=int,  default=256)
    parser.add_argument("--l1_iterations", type=int,  default=100000)

    parser.add_argument("--datapath", type=str,  default="../data/CIFAR10")
    parser.add_argument("-I","--ignore_trained", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    train()
