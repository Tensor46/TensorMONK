# TensorMONK

A collection of deep learning architectures (a PyTorch implementation).

## Dependencies
* python >= 3.6
* PyTorch > 0.4.1
* torchvision
* visdom

## How to train ImageNet?

If you have more nvidia cards & cores available, adjust the batch size (BSZ), number of GPUs (gpus), & number of threads (cpus) accordingly in the ./ImageNet.sh. Next, select an available architecture and update your train & validation folder location (trainDataPath and testDataPath). Finally, run ./ImageNet.sh.

## How to train CapsuleNet?

To replicate Hinton's paper on MNIST, run the following:

python Capsule.py -A capsule -B 256 -E 500 --optimizer adam --gpus 2 --cpus 6 --trainDataPath ./data --testDataPath ./data --replicate_paper

Ignore the replicate_paper argument to create a deep architecture (with few residual blocks before primary capsule). You can essentially add any block available in NeuralLayers to create a deeper architecture, which is followed by a primary capsule and secondary capsule. However, do consider two things 1. if you do reconstruction, update the reconstruction network relative to tensor_size, 2. capsule nets do require a good amount of gpu ram.

## Generative Adversarial Networks GAN

### [Progressive Growing of GANs](https://arxiv.org/pdf/1710.10196.pdf)

Trained on CIFAR10 (pggan-cifar10.py) -- requires more training (more gpus)!

![Generator at 4x4](https://github.com/Tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level1.gif)
![Generator at 8x8](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level2.gif)
![Generator at 16x16](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level3.gif)
![Generator at 32x32](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level4.gif)

## Details on tensormonk
### activations
* Activations: A supporting function for tensormonk.layers.Convolution. Available options (elu/lklu/[maxo](https://arxiv.org/pdf/1302.4389.pdf)/prelu/relu/relu6/rmxo(relu + maxo)/sigm/[squash](https://arxiv.org/pdf/1710.09829.pdf)/[swish](https://arxiv.org/pdf/1710.05941v1.pdf)/tanh)

### architectures
* [CapsuleNet](https://arxiv.org/pdf/1710.09829.pdf) (Hinton's version, and an example deep network)
* [ContextNet](https://arxiv.org/pdf/1805.04554.pdf)
* [ConvolutionalVAE](https://arxiv.org/pdf/1312.6114v10.pdf)
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) -- use type (see Table 1 in paper) - d121/d169/d201/d264. Pretrained weights are available for d121, d169, & d201.
* [InceptionV4](https://arxiv.org/pdf/1602.07261.pdf)
* [LinearVAE](https://arxiv.org/pdf/1312.6114v10.pdf)
* [MobileNetV1](https://arxiv.org/pdf/1704.04861.pdf)
* [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf)
* [PGGAN](https://arxiv.org/pdf/1710.10196.pdf)
* [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Discriminative_and_CVPR_2017_paper.pdf)
* ResidualNet -- use type = [r18/r34/r50/r101/r152](https://arxiv.org/pdf/1512.03385.pdf) or [rn50/rn101/rn152 for ResNeXt](https://arxiv.org/pdf/1611.05431.pdf) or [ser50/ser101/ser152 for Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) or sern50/sern101/sern152 (ResNeXt + Squeeze-and-Excitation Networks). Pretrained weights are available for r18, r34, r50, r101, & r152.
* [ShuffleNet](https://arxiv.org/pdf/1707.01083.pdf) (type = g1, g2, g3, g4, g8 from table 1)
* SimpleNet
* [NeuralDecisionForest](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)
* [UNet](https://arxiv.org/pdf/1505.04597.pdf)
  * UNetPatch: Works on small patches.

### data
* DataSets: (MNIST/FashionMNIST/CIFAR10/CIFAR100)
* FewPerLabel: Folder iterator to sample n consecutive samples per label
* FolderITTR: A wrapper on torchvision image folder iterator

### essentials
* CudaModel: Converts any model (pytorch module/sequential) to run on single/multiple gpu or cpu
* LoadModel: Loads pre-trained models with CudaModel
* MakeModel: Builds model using base class
* SaveModel: Save models (state_dict of anything in CudaModel (nn.Module) that starts with net in BaseModel, and rest as is)

### layers
* Convolution: A convolution/convolution transpose layer with dropout, normalization and activation
* ConvolutionalSAE: Base block for Stacked Convolutional Auto-encoder
* Linear: A linear layer with dropout and activation
* ResidualOriginal: Base residual block for ResNet18 and ResNet34
* ResidualComplex: Bottleneck residual block for ResNet50, ResNet101 and ResNet152
* ResidualInverted: Base block for MobileNetV2
* ResidualShuffle: Base block for ShuffleNet
* ResidualNeXt: Bottleneck residual block for ResNeXt
* SEResidualComplex: Bottleneck residual block for Squeeze and Excitation Networks
* SEResidualNeXt: Bottleneck residual block to combined ResNeXt, and Squeeze & Excitation Networks
* SimpleFire: Base block for SqueezeNet
* CarryModular: Supporting block for several modules
* DenseBlock: Base block for DenseNet
* Stem2: For InceptionV4
* InceptionA: For InceptionV4
* InceptionB: For InceptionV4
* InceptionC: For InceptionV4
* ReductionA: For InceptionV4
* ReductionB: For InceptionV4
* ContextNet_Bottleneck: Base block for ContextNet
* PrimaryCapsule: Primary Capsule for Dynamic routing between capsules
* RoutingCapsule: Routing Capsule for Dynamic routing between capsules
* DetailPooling: Detailed pooling layer
* DoG: Computes difference of two blurred tensors with different gaussian kernels
* DoGBlob: Aggregates several DoG's
* GaussianBlur: Creates Gaussian kernel given sigma and width. n_stds is fixed to 3.
* DoH: Computes determinant of Hessian of BCHW torch.Tensor using the cornor pixels of widthxwidth patch
* HessianBlob: Aggregates determinant of Hessian with width ranging from min_width to max_width (skips every other).

### loss
* [CapsuleLoss](https://arxiv.org/pdf/1710.09829.pdf)
* Categorical: Cross entropy / softmax / [taylor softmax](https://arxiv.org/pdf/1511.05042.pdf) / [large margin cosine loss](https://arxiv.org/pdf/1801.09414.pdf) / [large-margin Gaussian Mixture](https://arxiv.org/pdf/1803.02988.pdf)
* [DiceLoss / Tversky Loss](https://arxiv.org/abs/1706.05721): Segmentation loss function
* TripletLoss: with soft and hard negative mining

### normalizations
* Normalizations: A supporting function for tensormonk.layers.Convolution. Available options (batch/group/instance/layer/pixelwise)

### optimizers

### plots
* make_gifs: Makes a gif using a list of images
* VisPlots: Visdom plots to monitor weights (histograms and 2D kernels larger than 3x3), and responses

### regularizations
* DropOut: Using tensor_size returns a dropout for linear layer, and 2D-dropout/dropblock for convolutional modules

### thirdparty
* LSUVinit: Does LSUV initialization

### utils
* ImageNetNorm: nn.Module that does ImageNet normalization
* corr_1d: Computes row wise correlation between two 2D torch.Tensor's of same shape
* xcorr_1d: Computes cross correlation of 2D torch.Tensor's of shape MxN, i.e, M vectors of length N
* roc: Computes receiver under operating curve for a given combination of (genuine and impostor) or (score matrix and labels)
