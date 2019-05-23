""" Training a Tiny-SSD320 / MobileNetV2SSD320 """

from __future__ import print_function, division
import argparse
import tensormonk
from tensormonk.essentials import BaseNetwork, BaseOptimizer, EasyTrainer
from tensormonk.utils import SSDUtils

# the configuration for SSD 320 - works for MobileNetV2SSD320 & TinySSD320
ratios1 = (1, 2, 1/2)
ratios2 = (1, 2, 3, 1/2, 1/3)
layer_infos = [
    SSDUtils.LayerInfo((None, None, 40, 40), ratios1, .10, 0.20),
    SSDUtils.LayerInfo((None, None, 20, 20), ratios2, .20, 0.37),
    SSDUtils.LayerInfo((None, None, 10, 10), ratios2, .37, 0.54),
    SSDUtils.LayerInfo((None, None,  5,  5), ratios2, .54, 0.71),
    SSDUtils.LayerInfo((None, None,  3,  3), ratios1, .71, 0.88),
    SSDUtils.LayerInfo((None, None,  1,  1), ratios1, .88, 1.05)]

CONFIG_SSD320 = {"model": "SSD320",
                 "tensor_size": (1, 3, 320, 320),
                 "n_labels": 21,
                 "layer_infos": layer_infos,
                 "boxes_per_layer": [4, 6, 6, 6, 4, 4],
                 "gcxcywh_var1": 0.1,
                 "gcxcywh_var2": 0.2,
                 "encode_iou_threshold": 0.5,
                 "detect_iou_threshold": 0.2,
                 "detect_score_threshold": 0.2,
                 "detect_top_n": 1,
                 "detect_n_objects": 50}


def parse_args():
    parser = argparse.ArgumentParser(description="SimpleMNIST")
    parser.add_argument("-B", "--BSZ", type=int, default=32)
    parser.add_argument("-E", "--Epochs", type=int, default=6)

    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--default_gpu", type=int,  default=0)
    parser.add_argument("--gpus", type=int,  default=1)
    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


class MyModel(EasyTrainer):

    def step(self, inputs, training):
        tensor, target_gcxcywh_boxes, targets = inputs
        if self.is_cuda:
            tensor = tensor.cuda()
            target_gcxcywh_boxes = target_gcxcywh_boxes.cuda()
            targets = targets.cuda()
        if not training:
            tensor.requires_grad_(False)
            target_gcxcywh_boxes.requires_grad_(False)
            targets.requires_grad_(False)

        self.model_container["detector"].zero_grad()
        self.model_container["loss"].zero_grad()
        gcxcywh_boxes, predictions = self.model_container["detector"](tensor)
        loss = self.model_container["loss"](gcxcywh_boxes, predictions,
                                            target_gcxcywh_boxes, targets)

        if training:
            loss.backward()
            self.optimizer.step()
            self.meter_container["loss"].update(loss)

            # visualize every n_visplots
            if not (self.iteration % self.n_visplots):
                params = []
                name = self.logs_name + "_grads.png"
                for n in self.model_container.keys():
                    params += list(
                        self.model_container[n].named_parameters())
                self.visplots.show_gradients(params, name)
            return {"monitor": ["loss"]}

        else:
            pass


if __name__ == '__main__':
    r""" An example to train SSD320 on PascalVOC2012 """
    args = parse_args()
    from tensormonk.data import DataSets

    # dataset
    trData, _, teData, n_labels, _ = DataSets(
        "PascalVOC2012", tensor_size=CONFIG_SSD320["tensor_size"],
        n_samples=args.BSZ)
    CONFIG_SSD320["n_labels"] = n_labels

    # define networks
    from tensormonk.architectures import TinySSD320
    translator = SSDUtils.Translator(**CONFIG_SSD320)
    detector = BaseNetwork(network=TinySSD320,
                           optimizer=None,
                           arguments={"translator": translator,
                                      **CONFIG_SSD320})
    loss_net = BaseNetwork(network=tensormonk.loss.MultiBoxLoss,
                           optimizer=None,
                           arguments={"translator": translator,
                                      **CONFIG_SSD320})

    model = MyModel(name="tinyssd320",
                    path="./models",
                    networks={"detector": detector, "loss": loss_net},
                    optimizer=BaseOptimizer(args.optimizer,
                                            {"lr": args.learningRate}),
                    transformations=None,
                    meters=["loss", "top1", "top5", "test_top1", "test_top5"],
                    n_checkpoint=-1,
                    default_gpu=args.default_gpu,
                    gpus=args.gpus,
                    ignore_trained=args.ignore_trained,
                    visplots=True,
                    n_visplots=100)
    model.train(trData, epochs=args.Epochs)
