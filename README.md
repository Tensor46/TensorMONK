# TensorMONK

A collection of deep learning architectures with flexible modules (a PyTorch implementation).

## Available Architectures
* ResidualNet (18/34/50/101/152)
* MobileNetV1
* MobileNetV2
* ShuffleNet (groups = 1/2/3/4/8)
* CapsuleNet (Hinton's version and an example deep network)


## Dependencies
* python 3.6
* PyTorch > 0.4
* torchvision

## ImageNet :: How to train?

If you have more nvidia graphic cards & cores available, adjust the batch size (BSZ), number of GPUs (gpus) & number of threads (cpus) accordingly.

python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.1 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.01 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.001 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
