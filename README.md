# TensorMONK

A collection of deep learning architectures with flexible modules (a PyTorch implementation).


## Details on core (NeuralArchitectures, NeuralEssentials, NeuralLayers)

### NeuralArchitectures
* [ResidualNet (18/34/50/101/152)](https://arxiv.org/pdf/1512.03385.pdf)
* [MobileNetV1](https://arxiv.org/pdf/1707.01083.pdf)
* [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
* [ShuffleNet (type = g1, g2, g3, g4, g8 from table 1)](https://arxiv.org/pdf/1707.01083.pdf)
* [CapsuleNet (Hinton's version, and an example deep network)](https://arxiv.org/pdf/1710.09829.pdf)

## Available Architectures
* ResidualNet (18/34/50/101/152)
* MobileNetV1
* MobileNetV2
* ShuffleNet (groups = 1/2/3/4/8)
* CapsuleNet (Hinton's version and an example deep network)


## Dependencies
* python 3.6
* PyTorch >= 0.4
* torchvision

## ImageNet :: How to train?

If you have more nvidia graphic cards & cores available, adjust the batch size (BSZ), number of GPUs (gpus) & number of threads (cpus) accordingly in the ./ImageNet.sh.
Further, update your train and validation folder location (trainDataPath and testDataPath). Finally, run ./ImageNet.sh.

## CapsuleNet :: How to train?

To replicate Hinton's paper on MNIST, run the following:

python Capsule.py -A capsule -B 256 -E 500 --optimizer adam --gpus 2 --cpus 6 --trainDataPath ./data --testDataPath ./data --replicate_paper

Ignore the replicate_paper argument to create a deep architecture (with few residual blocks before primary capsule). You can essentially add any blocks available in NeuralLayers to create a deep architecture, followed by a primary capsule and secondary capsule on any dataset. However, do consider two things 1. if you do reconstruction, update the reconstruction network accordingly, 2. capsule nets do require a good amount of gpu ram.
