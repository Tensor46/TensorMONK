""" Training a 3 layer cnn on mnist and fashion mnist using EasyTrainer """

from __future__ import print_function, division
import argparse
import tensormonk
from tensormonk.essentials import BaseNetwork, BaseOptimizer, EasyTrainer


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

    parser.add_argument("-I", "--ignore_trained", action="store_true")

    return parser.parse_args()


class MyModel(EasyTrainer):

    def step(self, inputs, training):
        tensor, target = inputs
        if self.is_cuda:
            tensor, target = tensor.cuda(), target.cuda()
        if hasattr(self, "transformations"):
            # transformations
            tensor = self.transformations(tensor)
            tensor.requires_grad_(True)
        if not training:
            tensor.requires_grad_(False)
            target.requires_grad_(False)

        self.model_container["embedding"].zero_grad()
        self.model_container["loss"].zero_grad()
        embedding = self.model_container["embedding"](tensor)
        loss, (top1, top5) = self.model_container["loss"](embedding, target)

        if training:
            loss.backward()
            self.optimizer.step()
            self.meter_container["loss"].update(loss)
            self.meter_container["top1"].update(top1)
            self.meter_container["top5"].update(top5)

            # visualize every n_visplots
            if not (self.iteration % self.n_visplots):
                params = []
                name = self.logs_name + "_grads.png"
                for n in self.model_container.keys():
                    params += list(
                        self.model_container[n].named_parameters())
                self.visplots.show_gradients(params, name)
            return {"monitor": ["loss", "top1", "top5"]}

        else:
            self.meter_container["test_top1"].update(top1)
            self.meter_container["test_top5"].update(top5)
            return {"monitor": ["test_top1", "test_top5"]}


if __name__ == '__main__':
    r""" An example to train 3 layer cnn on mnist and fashion mnist """
    args = parse_args()
    from tensormonk.data import DataSets
    trData, vaData, teData, n_labels, tensor_size = \
        DataSets(args.dataset, data_path="../data", n_samples=args.BSZ)
    embedding_net = BaseNetwork(network=tensormonk.architectures.SimpleNet,
                                optimizer=None,
                                arguments={"tensor_size": (1, 1, 28, 28)})
    loss_net = BaseNetwork(network=tensormonk.loss.Categorical,
                           optimizer=None,
                           arguments={"tensor_size": (1, 64),
                                      "n_labels": n_labels,
                                      "type": args.loss_type,
                                      "measure": args.loss_measure})

    from tensormonk.data import ElasticSimilarity, RandomTransforms
    transformations = BaseNetwork(
        network=RandomTransforms,
        arguments={"functions": [ElasticSimilarity()], "probabilities": [0.5]})

    model = MyModel(name="simplenet",
                    path="./models",
                    networks={"embedding": embedding_net, "loss": loss_net},
                    optimizer=BaseOptimizer(args.optimizer,
                                            {"lr": args.learningRate}),
                    transformations=transformations,
                    meters=["loss", "top1", "top5", "test_top1", "test_top5"],
                    n_checkpoint=-1,
                    default_gpu=args.default_gpu,
                    gpus=args.gpus,
                    ignore_trained=args.ignore_trained,
                    visplots=True,
                    n_visplots=100)
    model.train(trData, teData, epochs=args.Epochs)
