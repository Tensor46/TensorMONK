""" Training a ESRGAN """

from __future__ import print_function, division
import argparse
import tensormonk
import torch
import torch.nn.functional as F
from tensormonk.loss import AdversarialLoss
from tensormonk.essentials import BaseNetwork, BaseOptimizer, EasyTrainer
SCHEDULER1 = SCHEDULER2 = None


def parse_args():
    parser = argparse.ArgumentParser(description="ESRGAN")
    parser.add_argument("-B", "--BSZ", type=int, default=16)
    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("-I", "--ignore_trained", action="store_true")
    return parser.parse_args()


class MyModel(EasyTrainer):

    def step(self, inputs, training):
        hr_tensor, lr_tensor = inputs
        if self.is_cuda:
            hr_tensor, lr_tensor = hr_tensor.cuda(), lr_tensor.cuda()

        if self.kwargs["phase1"]:
            return self.phase1(hr_tensor, lr_tensor)
        else:
            return self.phase2(hr_tensor, lr_tensor)

    def phase1(self, hr: torch.Tensor, lr: torch.Tensor):
        # only generator is trained
        self.model_container["generator"].zero_grad()
        g_of_lr = self.model_container["generator"](lr)
        loss_l1 = F.smooth_l1_loss(g_of_lr, hr)
        loss_l1.backward()
        self.optim_container["generator"].step()
        if SCHEDULER1 is not None:
            SCHEDULER1.step()
        self.meter_container["loss_l1"].update(loss_l1)
        if not (self.iteration % self.n_visplots):
            x = torch.cat((hr[:4], g_of_lr[:4]))
            self.visplots.visplots.images(
                        (x * 0.25 + 0.5).clamp(0, 1), nrow=2,
                        opts={"title": "SR"}, win="SR")
        return {"monitor": ["loss_l1"]}

    def phase2(self, hr: torch.Tensor, lr: torch.Tensor):
        # both generator and discriminator are trained

        # training generator
        self.model_container["generator"].zero_grad()
        g_of_lr = self.model_container["generator"](lr)
        loss_l1 = F.smooth_l1_loss(g_of_lr, hr)
        loss_pr = F.smooth_l1_loss(self.model_container["vgg19"](g_of_lr),
                                   self.model_container["vgg19"](hr))
        loss_g = AdversarialLoss.g_relativistic(
            self.model_container["discriminator"](hr),
            self.model_container["discriminator"](g_of_lr))
        loss = loss_pr + 0.005 * loss_g + 0.01 * loss_l1
        loss.backward()
        self.optim_container["generator"].step()
        if SCHEDULER1 is not None:
            SCHEDULER1.step()
        self.meter_container["loss_pr"].update(loss_pr)
        self.meter_container["loss_l1"].update(loss_l1)
        self.meter_container["loss_g"].update(loss_g)
        self.meter_container["loss"].update(loss)
        if not (self.iteration % self.n_visplots):
            x = torch.cat((hr[:4], g_of_lr[:4]))
            self.visplots.visplots.images(
                        (x * 0.25 + 0.5).clamp(0, 1), nrow=2,
                        opts={"title": "SR"}, win="SR")

        # training discriminator
        self.model_container["discriminator"].zero_grad()
        loss_d = AdversarialLoss.d_relativistic(
            self.model_container["discriminator"](hr.detach()),
            self.model_container["discriminator"](g_of_lr.detach()))
        loss_d.backward()
        self.optim_container["discriminator"].step()
        if SCHEDULER2 is not None:
            SCHEDULER2.step()
        self.meter_container["loss_d"].update(loss_d)
        return {"monitor": ["loss", "loss_pr", "loss_l1", "loss_g", "loss_d"]}


if __name__ == '__main__':
    r""" An example to train ESRGAN """
    args = parse_args()

    # dataset
    tensor_size: tuple = (1, 3, 32, 32)
    n_upscale: int = 2

    trData = tensormonk.data.SuperResolutionData(
        path="../data/sr_data",
        t_size=tensor_size,
        n_upscale=n_upscale,
        test=False,
        add_flickr2k=True)
    n_batches = len(trData) // args.BSZ
    trData = torch.utils.data.DataLoader(
        trData,
        batch_size=args.BSZ,
        shuffle=True,
        num_workers=8)

    # define networks
    print("... training phase1")
    generator = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.Generator,
        optimizer=BaseOptimizer("adam", {"lr": 0.0002}))
    discriminator = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.Discriminator,
        optimizer=BaseOptimizer("adam", {"lr": 0.0002}))
    vgg19 = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.VGG19,
        optimizer=None,
        only_eval=True)

    model = MyModel(name="esrgan",
                    path="./models",
                    networks={"generator": generator,
                              "discriminator": discriminator,
                              "vgg19": vgg19},
                    optimizer=None,
                    meters=["loss_pr", "loss_l1", "loss_g", "loss_d"],
                    n_checkpoint=1000,
                    default_gpu=args.default_gpu,
                    gpus=args.gpus,
                    ignore_trained=args.ignore_trained,
                    visplots=True,
                    n_visplots=100,
                    phase1=True)
    SCHEDULER1 = torch.optim.lr_scheduler.MultiStepLR(
        model.optim_container["generator"], [50000, 100000], gamma=0.5)
    model.train(trData, epochs=150000 // n_batches)

    print("... training phase2")
    generator = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.Generator,
        optimizer=BaseOptimizer("adam", {"lr": 0.0001}))
    discriminator = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.Discriminator,
        optimizer=BaseOptimizer("adam", {"lr": 0.0001}))
    vgg19 = BaseNetwork(
        network=tensormonk.architectures.ESRGAN.VGG19,
        optimizer=None,
        only_eval=True)

    model = MyModel(name="esrgan",
                    path="./models",
                    networks={"generator": generator,
                              "discriminator": discriminator,
                              "vgg19": vgg19},
                    optimizer=None,
                    meters=["loss", "loss_pr", "loss_l1", "loss_g", "loss_d"],
                    n_checkpoint=1000,
                    default_gpu=args.default_gpu,
                    gpus=args.gpus,
                    ignore_trained=args.ignore_trained,
                    visplots=True,
                    n_visplots=100,
                    phase1=False)

    steps = [50000, 100000, 200000, 300000]
    SCHEDULER1 = torch.optim.lr_scheduler.MultiStepLR(
        model.optim_container["generator"], steps, gamma=0.5)
    SCHEDULER2 = torch.optim.lr_scheduler.MultiStepLR(
        model.optim_container["discriminator"], steps, gamma=0.5)
    model.train(trData, epochs=300000 // n_batches)
