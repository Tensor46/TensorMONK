""" TensorMONK's :: essentials """

import os
import numpy as np
import torch
import torch.nn as nn
from .utils import Meter
from ..plots import VisPlots
from collections import OrderedDict


class BaseOptimizer:
    r"""BaseOptimizer class that contains optimizer type, and related arguments.

    Args:
        type (required, str): Optimizer type, options = SGD/Adam. Default = sgd
        arguments (optional, dict): All the arguments required to build the
            optimizer. Default = {}
            When arguments={} and type="sgd",
                lr = 0.1
            When arguments={} and type="adam",
                lr = 0.001
                betas = (0.9, 0.999)
                eps = 1e-8
                weight_decay = 0
                amsgrad = True

    Ex:
        optimizer = BaseOptimizer(type="sgd", arguments={"lr": 0.1,
            "momentum": 0.9, "weight_decay": 0.00005})
    """
    def __init__(self, type: str = "sgd", arguments: dict = {}):

        if type.lower() == "sgd":
            self.algorithm = torch.optim.SGD
            if len(arguments) == 0 or "lr" not in arguments.keys():
                arguments["lr"] = 0.1

        elif type.lower() == "adam":
            self.algorithm = torch.optim.Adam
            if len(arguments) == 0 or "lr" not in arguments.keys():
                arguments["lr"] = 0.001

        else:
            raise NotImplementedError
        self.arguments = arguments


class BaseNetwork:
    r"""BaseNetwork class that contains the network (nn.Module object),
    arguments required for the network, optimizer, default_gpu, gpus,
    ignore_trained, and only_eval. (optimizer, default_gpu, gpus,
    ignore_trained, and only_eval) can overwrite the EasyTrainer arguments.

    Args:
        network (required, torch.nn.Module): An nn.Module that defines the
            network.
        arguments (optional, dict): All the arguments required to initialize
            the network. Default = {}
        optimizer (optional, BaseOptimizer): Only required when the network
            requires a different optimizer (ex: a generator and discriminator
            in a GAN) than the global optimizer provided to EasyTrainer. Look
            into BaseOptimizer for more information. Default = None
        default_gpu (optional, int): The default gpu to train. When None, uses
            the default_gpu in EasyTrainer. default = None
        gpus (optional, int): Number of gpus used to train. When None, uses
            the gpus in EasyTrainer. default = None
        ignore_trained (optional, bool): Ignores any pretrained model. When
            None, uses the ignore_trained value provided to EasyTrainer.
            default = None
        only_eval (optional, bool): When True, forces the network to be in
            eval mode.

    Ex:
        embedding = BaseNetwork(network=tensormonk.architectures.SimpleNet,
                                optimizer=None,
                                arguments={"tensor_size": (1, 1, 28, 28)})
    """
    def __init__(self,
                 network: torch.nn.Module,
                 arguments: dict = {},
                 optimizer: BaseOptimizer = None,
                 default_gpu: int = None,
                 gpus: int = None,
                 ignore_trained: bool = None,
                 only_eval: bool = False):
        self.network = network
        self.arguments = arguments
        self.optimizer = optimizer
        self.default_gpu = default_gpu
        self.gpus = gpus
        self.ignore_trained = ignore_trained
        self.only_eval = only_eval


class EasyTrainer(object):
    r"""A trainer object scalable to several (cnn/gan/rnn) models. The step has
    to be defined for the model that has to be trained. There is no limit on
    "networks", as in, for GANs, one can define a generator and a discriminator
    in networks (a dictonary of BaseNetwork's). All the "networks" will be
    built in model_container (OrderedDict). All the arguments in BaseNetwork
    can overwrite the arguments provided to EasyTrainer (each network can have
    its own optimizer, default_gpu, gpus, ignore_trained and only_eval) -- one
    can have a generator on gpu-0 and discriminator on gpu-1. "optimizer" must
    be a BaseOptimizer object, which accumulates all the parameters in the
    networks when networks[n].optimizer==None and networks[n].only_eval==False.

    Args:
        name (required, str): Name of the model. A folder is created to save
            the model and any other outputs (from VisPlots). Assigns a
            file_name (where the model is saved) and logs_name (that is used as
            a prefix for VisPlots or any other logs).
        path (required, str): path in which a folder with name is created.
        networks (required, dict): a dictionary of BaseNetwork's. The keys are
            used to build all the networks in model_container
            (CPU/GPU/DataParallel-GPU).
        optimizer (optional, BaseOptimizer): A global optimizer that
            accumulates all the parameters in the model_container (if
            networks[n].optimizer==None and networks[n].only_eval==False).
        transformations (optional, BaseNetwork): Exclusive for
            tensormonk.data.RandomTransforms that can scale on GPUs
        meters (optional, list): list of all the meters that need to be
            monitored. Assigns a Meter object to all the list (must be updated
            in the step).
        n_checkpoint (optional, list): Used to save a checkpoint and test
            when teData is provided. When > 0, saves at every n_checkpoint
            iterations. When -1, saves after every epoch. default = 2000
        default_gpu (optional, int): The default gpu to train, ignored when
            gpus > 1 or (networks[n].default_gpu or networks[n].gpus) is
            provided. default = 0
        gpus (optional, int): Number of gpus used to train, ignored when
            (networks[n].default_gpu or networks[n].gpus) is provided.
            default = 1
        ignore_trained (optional, bool): Ignores any pretrained model.
            default = False
        visplots (optional, bool): When True, enables tensormonk.plots.VisPlots
        n_visplots (optional, int):

    Ex:
        import tensormonk
        from tensormonk.data import DataSets
        from tensormonk.essentials import BaseNetwork, BaseOptimizer
        trData, vaData, teData, n_labels, tensor_size = \
            DataSets("mnist", data_path="../data", n_samples=32)
        embedding = BaseNetwork(network=tensormonk.architectures.SimpleNet,
                                optimizer=None,
                                arguments={"tensor_size": (1, 1, 28, 28)})
        loss = BaseNetwork(network=tensormonk.loss.Categorical,
                           optimizer=None,
                           arguments={"tensor_size": (1, 64), "n_labels": 10})
        model = MyModel(name="simplenet", path="./models",
                        networks={"embedding": embedding, "loss": loss},
                        meters=["loss", "top1", "top5",
                                "test_top1", "test_top5"],
                        optimizer=BaseOptimizer(), n_checkpoint=-1,
                        ignore_trained=True)
        model.train(trData, teData, epochs=6)

    """
    def __init__(self,
                 name: str,
                 path: str,
                 networks: dict,
                 optimizer: BaseOptimizer,
                 transformations: BaseNetwork = None,
                 meters: list = ["loss"],
                 n_checkpoint: int = 2000,
                 default_gpu: int = 0,
                 gpus: int = 1,
                 ignore_trained: bool = False,
                 visplots: bool = False,
                 n_visplots: int = 100):

        # checks
        if not isinstance(n_checkpoint, int):
            raise TypeError("EasyTrainer: n_checkpoint must be int: "
                            "{}".format(type(n_checkpoint).__name__))
        if not (n_checkpoint >= -1):
            raise ValueError("EasyTrainer: n_checkpoint must be >= -1: "
                             "{}".format(n_checkpoint))
        if not isinstance(default_gpu, int):
            raise TypeError("EasyTrainer: default_gpu must be int: "
                            "{}".format(type(default_gpu).__name__))
        if not isinstance(gpus, int):
            raise TypeError("EasyTrainer: gpus must be int: "
                            "{}".format(type(gpus).__name__))
        if not (gpus >= 0):
            raise ValueError("EasyTrainer: gpus must be >= 0: {}".format(gpus))
        if not isinstance(ignore_trained, bool):
            raise TypeError("EasyTrainer: ignore_trained must be bool: "
                            "{}".format(type(ignore_trained).__name__))
        if not isinstance(visplots, bool):
            raise TypeError("EasyTrainer: visplots must be bool: "
                            "{}".format(type(visplots).__name__))
        if not isinstance(n_visplots, int):
            raise TypeError("EasyTrainer: n_visplots must be int: "
                            "{}".format(type(n_visplots).__name__))
        if not (n_visplots >= 1):
            raise ValueError("EasyTrainer: n_visplots must be >= 1: "
                             "{}".format(n_visplots))

        self.is_cuda = torch.cuda.is_available()
        self.default_gpu = default_gpu
        self.gpus = gpus
        self.n_checkpoint = n_checkpoint
        self.ignore_trained = ignore_trained
        self.iteration = 0
        self.n_visplots = n_visplots

        self._check_path(name, path)
        self._check_networks(networks)
        self._check_optimizer(optimizer, networks)
        self.model_container = OrderedDict()
        self.meter_container = OrderedDict()
        self.optim_container = OrderedDict()
        self._build_networks(networks)
        self._build_optimizers(optimizer, networks)
        self._build_meters(meters)
        self._build_transformations(transformations)
        if visplots:
            self.visplots = VisPlots(self.name)

    def train(self, train_data, test_data=None, epochs=1, **kwargs):
        for epoch in range(epochs):
            for i, inputs in enumerate(train_data):
                output = self.step(inputs, training=True)
                if "monitor" in output.keys():
                    print(" ... " +
                          self._monitor(output["monitor"]), end="\r")
                self.iteration += 1

                # save the model is n_checkpoint > 0
                if self.n_checkpoint > 0 and \
                   not (self.iteration % self.n_checkpoint):
                    self._save()
                    if "monitor" in output.keys():
                        print(" ... " + self._monitor(output["monitor"],
                                                      self.n_checkpoint))
                    if test_data is not None:
                        self.test(test_data)

            # save the model every epoch (n_checkpoint = -1)
            if self.n_checkpoint == -1:
                self._save()
                if "monitor" in output.keys():
                    print(" ... " + self._monitor(output["monitor"], i))
                if test_data is not None:
                    self.test(test_data)

    def test(self, test_data):
        current_states = []
        # check models in eval mode and convert everything to eval()
        for n in self.model_container.keys():
            current_states += [self.model_container[n].training]
            self.model_container[n].eval()
        # testing using step
        for i, inputs in enumerate(test_data):
            output = self.step(inputs, training=False)
            if "monitor" in output.keys():
                print(" ... test "+self._monitor(output["monitor"]), end="\r")
        if "monitor" in output.keys():
            print(" ... test " + self._monitor(output["monitor"], i))
        # convert all trainable models from eval to train
        for value, n in zip(current_states, self.model_container.keys()):
            if value:
                self.model_container[n].train()

    def step(self, inputs, training):
        r"""Define what need to be done. "training" is True when called from
        train, and False when called from test """
        pass

    def _monitor(self, tags, n=1):
        r"""Convert a list of tags in meter_container to string"""
        msg = ["{} {:3.2f}".format(x, self.meter_container[x].average(n))
               for x in tags if x in self.meter_container.keys()]
        return " :: ".join(["{:6d}".format(self.iteration)] + msg) + (" "*6)

    def _check_path(self, name, path):
        r"""Check the path & name, and create a folder to save the model &
        related files. Checks if there is a pretrained model and sets the
        is_pretrained variable.
        """
        if not isinstance(name, str):
            raise TypeError("EasyTrainer: name must be str: "
                            "{}".format(type(name).__name__))
        if not isinstance(path, str):
            raise TypeError("EasyTrainer: path must be str: "
                            "{}".format(type(path).__name__))
        if not os.path.isdir(path):
            raise ValueError("EasyTrainer: path is not a valid directory: "
                             "{}".format(path))
        self.path = os.path.join(path, name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.name = name
        self.file_name = os.path.join(self.path, name + ".t7")
        self.logs_name = os.path.join(self.path, name)
        self.is_pretrained = False
        if os.path.isfile(self.file_name):
            self.is_pretrained = True
        return None

    def _check_networks(self, networks):
        r"""Check if the "networks" is a dictonary, and all the values are
        BaseNetwork.
        """
        if not isinstance(networks, dict):
            raise TypeError("EasyTrainer: networks must be dict: "
                            "{}".format(type(networks).__name__))
        network_names = list(networks.keys())
        if len(network_names) == 0:
            raise ValueError("EasyTrainer: networks is an empty dict")
        if any(map(lambda x: not isinstance(networks[x], BaseNetwork),
                   network_names)):
            raise TypeError("EasyTrainer: one/more values in networks are not "
                            "BaseNetwork")
        return None

    def _check_optimizer(self, optimizer, networks):
        r"""Check if the "optimizer" and all networks[x].optimizer (when not
        None) are BaseOptimizer.
        """
        if optimizer is not None:
            if not isinstance(optimizer, BaseOptimizer):
                raise TypeError("EasyTrainer: optimizer must be BaseOptimizer:"
                                " {}".format(type(optimizer).__name__))
        for n in list(networks.keys()):
            if networks[n].optimizer is not None:
                if not isinstance(networks[n].optimizer, BaseOptimizer):
                    raise TypeError("EasyTrainer: {}'s optimizer ".format(n) +
                                    "is not BaseOptimizer")

    def _build_networks(self, networks):
        r"""Builds all networks and load pretrained weights if exists. The
        EasyTrainer params are overwritten by networks[network] parameters.
        """
        if self.is_pretrained:
            content = torch.load(self.file_name)["model_container"]
        for n in list(networks.keys()):
            _pretrained = False
            print("... building {}".format(n), end="\r")
            if isinstance(networks[n].network, torch.nn.Module):
                self.model_container[n] = networks[n].network
            else:
                self.model_container[n] = \
                    networks[n].network(**networks[n].arguments)

            # load pretrained
            if self.is_pretrained:
                ignore_trained = self.ignore_trained
                if networks[n].ignore_trained is not None:
                    # networks' parameters will overwrite EasyTrainer's
                    ignore_trained = networks[n].ignore_trained
                if not ignore_trained and n in content.keys():
                    if content[n] is not None:
                        self.model_container[n].load_state_dict(content[n])
                        _pretrained = True

            # cuda and DataParallel
            if networks[n].gpus is not None:
                gpus = networks[n].gpus
            else:
                gpus = self.gpus
            if networks[n].default_gpu is not None:
                default_gpu = networks[n].default_gpu
            else:
                default_gpu = self.default_gpu
            if self.is_cuda and gpus == 1:
                torch.cuda.set_device(default_gpu)
                self.model_container[n].cuda()
            if self.is_cuda and gpus > 1:
                _gpus = list(range(gpus))
                self.model_container[n] = \
                    nn.DataParallel(self.model_container[n], device_ids=_gpus)
            if networks[n].only_eval is not None:
                if networks[n].only_eval:
                    self.model_container[n].eval()
            n_params = np.sum([p.numel() for p in
                               self.model_container[n].parameters()])
            print("... Network {} has {} parameters".format(n, n_params) +
                  (" :: loaded pretrained weights" if _pretrained else ""))

    def _build_optimizers(self, optimizer, networks):
        r"""Builds all optimizer and networks.optimizer (if any) """

        all_params = []
        for n in self.model_container.keys():
            if not networks[n].only_eval:
                params = list(self.model_container[n].parameters())
                if isinstance(networks[n].optimizer, BaseOptimizer):
                    self.optim_container[n] = networks[n].optimizer.algorithm(
                        params, **networks[n].optimizer.arguments)
                else:
                    all_params += params

        if len(all_params) > 0 and isinstance(optimizer, BaseOptimizer):
            self.optimizer = optimizer.algorithm(all_params,
                                                 **optimizer.arguments)

    def _build_meters(self, meters):
        r"""Initilizes Meter object for all the meters and loads pretained!"""
        if len(meters) == 0:
            return
        if self.is_pretrained and not self.ignore_trained:
            content = torch.load(self.file_name)
            self.iteration = content["iteration"]

        for m in meters:
            self.meter_container[m] = Meter()
            if "content" in locals():
                self.meter_container[m].values = content["meter_container"][m]

    def _build_transformations(self, transformations):
        r""" Builds CPU/GPU pytorch based transformations (compatible module is
        tensormonk.data.RandomTransforms) """
        if isinstance(transformations, BaseNetwork):
            print("... building transformations", end="\r")
            if isinstance(transformations.network, torch.nn.Module):
                self.transformations = transformations.network
            else:
                self.transformations = \
                    transformations.network(**transformations.arguments)

            # cuda and DataParallel
            if transformations.gpus is not None:
                gpus = transformations.gpus
            else:
                gpus = self.gpus
            if transformations.default_gpu is not None:
                default_gpu = transformations.default_gpu
            else:
                default_gpu = self.default_gpu
            if self.is_cuda and gpus == 1:
                torch.cuda.set_device(default_gpu)
                self.transformations.cuda()
            if self.is_cuda and gpus > 1:
                _gpus = list(range(gpus))
                self.transformations = \
                    nn.DataParallel(self.transformations, device_ids=_gpus)
            self.transformations.eval()
            n_params = np.sum([p.numel() for p in
                               self.transformations.parameters()])
            print("... Transformations has {} parameters".format(n_params))

    def _save(self):
        r""" Saves the model_container and meter_container as dict given the
        file_name """
        content = {"model_container": {}, "meter_container": {},
                   "iteration": self.iteration}
        for key in self.model_container.keys():
            tmp = self.model_container[key].state_dict()
            content["model_container"][key] = self._cpu_state_dict(
                self._convert_state_dict(tmp))

        for key in self.meter_container.keys():
            content["meter_container"][key] = self.meter_container[key].values
        torch.save(content, self.file_name)

    @staticmethod
    def _convert_state_dict(state_dict):
        r""" Converts nn.DataParallel state_dict to nn.Module state_dict """
        new_state_dict = OrderedDict()
        for x in state_dict.keys():
            new_state_dict[x[7:] if x.startswith("module.") else x] = \
                state_dict[x]
        return new_state_dict

    @staticmethod
    def _cpu_state_dict(state_dict):
        r""" Converts all state_dict to cpu state_dict """
        new_state_dict = OrderedDict()
        for x in state_dict.keys():
            new_state_dict[x] = state_dict[x].data.detach().cpu()
        return new_state_dict
