"""ImageNet training."""


from __future__ import print_function, division
import sys
import math
import argparse
import torch
import torchvision as tv
import numpy as np
import tensormonk


class Trainer(tensormonk.essentials.EasyTrainer):
    """Trainer TensorMONK."""

    def step(self, inputs: tuple, training: bool) -> dict:
        """Define a step."""
        if torch.cuda.is_available() and self.gpus > 0:
            tensor, target = map(lambda x: x.cuda(), inputs)

        self.optimizer.zero_grad()
        predictions = self.model_container["network"](tensor)
        loss = self.model_container["loss"](predictions, target)
        top1, top5 = tensormonk.loss.utils.compute_top15(predictions, target)

        loss, top1, top5 = map(lambda x: x.mean() if x.ndim else x,
                               (loss, top1, top5))

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)

        if training:
            self.backward(loss, self.optimizer)
            if self.is_ddp:
                loss = self.ddp_handler.all_reduce(loss)
                top1 = self.ddp_handler.all_reduce(top1)
                top5 = self.ddp_handler.all_reduce(top5)

            self.meter_container["loss"].update(loss)
            self.meter_container["top1"].update(top1)
            self.meter_container["top5"].update(top5)
            return {"monitor": ["loss", "top1", "top5"]}

        self.meter_container["test_top1"].update(top1)
        self.meter_container["test_top5"].update(top5)
        return {"monitor": ["test_top1", "test_top5"]}

    def save_criteria(self) -> bool:
        """Save based on evaluator score."""
        if self.is_ddp and not self.ddp_handler.is_main_process:
            return False

        criteria = True
        values = self.meter_container["test_top1"].values
        if len(values) > 2 and values[-1] < max(values[:-1]):
            criteria = False

        if criteria:
            print("... saving.")
        return criteria


def main(rank: int, world_size: int, args: argparse.ArgumentParser):
    # ddp handler
    if rank is None or world_size is None or args.gpus == 1:
        is_ddp, ddp_handler = False, None
    else:
        ddp_handler = tensormonk.essentials.HandleDDP(rank, world_size)
        is_ddp: bool = ddp_handler.is_ddp

    # seed
    if args.seed is not None and isinstance(args.seed, int):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # training data
    tr_dataset = tv.datasets.ImageFolder(
        "/raid/imagenet/train",
        tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.25, 0.25, 0.25])]))

    va_dataset = None
    if (not is_ddp) or (is_ddp and ddp_handler.is_main_process):
        va_dataset = tv.datasets.ImageFolder(
            "/raid/imagenet/validation",
            tv.transforms.Compose([
                tv.transforms.Resize(256),
                tv.transforms.CenterCrop(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.25, 0.25, 0.25])]))

    # define networks and build model
    networks = {"network": tensormonk.essentials.BaseNetwork(
                    network=tv.models.resnet18, arguments={}),
                "loss": tensormonk.essentials.BaseNetwork(
                    network=torch.nn.CrossEntropyLoss, arguments={})}
    trainer = Trainer(
        name=f"resnet18_{args.seed}",
        path="./weights",
        networks=networks,
        optimizer=tensormonk.essentials.BaseOptimizer(
            "sgd", {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}),
        meters=["loss", "top1", "top5", "test_top1", "test_top5"],

        n_checkpoint=-1,
        default_gpu=(rank if is_ddp else args.default_gpu),
        gpus=(1 if is_ddp else args.gpus),
        ignore_trained=args.ignore_trained,
        visplots=True, n_visplots=100,
        precision="mixed",
        clip=None,
        monitor_ndigits=5,
        ddp_handler=ddp_handler)

    # prepare loader
    tr_loader = trainer.prepare_dataloader(
        tr_dataset,
        batch_size=args.batch_size, n_workers=None, shuffle=True)
    va_loader = None
    if va_dataset is not None:
        va_loader = trainer.prepare_dataloader(
            va_dataset,
            batch_size=args.batch_size, n_workers=None, shuffle=False,
            test_data=True)

    # train
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = 0.1
    trainer.train(tr_loader, va_loader, epochs=30)
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = 0.01
    trainer.train(tr_loader, va_loader, epochs=30)
    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = 0.001
    trainer.train(tr_loader, va_loader, epochs=30)


def parse_args():
    """TensorMONK imagenet DDP."""
    parser = argparse.ArgumentParser(description="TensorMONK imagenet DDP!")
    parser.add_argument("-b", "--batch_size",   type=int,   default=256)
    parser.add_argument("-lr", "--lr",          type=float, default=0.1)
    parser.add_argument("--default_gpu",        type=int,   default=0)
    parser.add_argument("--gpus",               type=int,   default=2)
    parser.add_argument("--seed",               type=int,   default=46)

    parser.add_argument("-ddp", "--distributed",  action="store_true")
    parser.add_argument("-i", "--ignore_trained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.distributed:
        try:
            world_size = args.gpus
            torch.multiprocessing.spawn(
                main,
                args=(world_size, args),
                nprocs=world_size,
                join=True)

        except KeyboardInterrupt:
            print("Interrupted")

        finally:
            torch.distributed.destroy_process_group()

    else:
        main(None, None, args)
