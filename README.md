# TensorMONK

A collection of deep learning architectures with flexible modules (a PyTorch implementation).


## Details on core (NeuralArchitectures, NeuralEssentials, NeuralLayers)

### NeuralArchitectures
* [ResidualNet (18/34/50/101/152)](https://arxiv.org/pdf/1512.03385.pdf)
* [MobileNetV1](https://arxiv.org/pdf/1704.04861.pdf)
* [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
* [ShuffleNet (type = g1, g2, g3, g4, g8 from table 1)](https://arxiv.org/pdf/1707.01083.pdf)
* [CapsuleNet (Hinton's version, and an example deep network)](https://arxiv.org/pdf/1710.09829.pdf)

### NeuralEssentials
* BaseModel -- A base class that contains networks (embedding, loss or any), meters (loss, accuracy etc), fileName, isCUDA
* CudaModel -- Converts any model (pytorch module) to run on single gpu's or multiple gpu's or cpu
* LoadModel -- Loads pretrained models (usually, from ./models)
* SaveModel -- Save models (usually, state_dict of anything that starts with net in BaseModel, and rest as is)

* FolderITTR -- PyTorch image folder iterator with few extras.
* MNIST -- To get MNIST, train and test dataset loader.

### NeuralLayers
#### Convolution -- A convolution layer with following parameters: 1. tensor_size = a list/tuple of length 4 (BxWxHxC - any B should work), 2. filter_size = int/list/tuple (if list/tuple, length must be 2), 3. out_channels = int, 4. strides = int/list/tuple (if list/tuple, length must be 2), 5. pad = True/False (True essentially delivers same output size when strides = 1, and False returns valid convolution). 6. activation = relu/relu6/lklu(leaky relu)/tanh/sigm/[maxo](https://arxiv.org/pdf/1302.4389.pdf)/[swish](https://arxiv.org/pdf/1710.05941v1.pdf), 7. dropout = 0. to 1. (adds dropout layer), 8. batch_nm = True/False (adds batch normalization when True), 9. pre_nm = True/False (when True along with batch_nm -- batch normalization + activation + convolution else convolution + batch normalization + activation), 10. groups = 1 (default), 11. weight_norm = [True](https://arxiv.org/pdf/1602.07868.pdf)/False
#### CarryResidue
*



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
