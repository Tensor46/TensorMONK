# TensorMONK

A collection of deep learning architectures (a PyTorch implementation).

## Dependencies
* python >= 3.6
* PyTorch > 1.0
* torchvision
* visdom

## Training models on 2012 ImageNet recognition task

If you have more nvidia cards & cores available, adjust the batch size (BSZ), number of GPUs (gpus), & number of threads (cpus) accordingly in the ./ImageNet.sh. Next, select an available architecture and update your train & validation folder location (trainDataPath and testDataPath). Finally, run ./ImageNet.sh.

## Training CapsuleNet on MNIST

To replicate Hinton's paper on MNIST, run the following:

python Capsule.py -A capsule -B 256 -E 500 --optimizer adam --gpus 2 --cpus 6 --trainDataPath ./data --testDataPath ./data --replicate_paper

Ignore the replicate_paper argument to create a deep architecture (with few residual blocks before primary capsule). You can essentially add any block available in NeuralLayers to create a deeper architecture, which is followed by a primary capsule and secondary capsule. However, do consider two things 1. if you do reconstruction, update the reconstruction network relative to tensor_size, 2. capsule nets do require a good amount of gpu ram.

## Generative Adversarial Networks

### [Progressive Growing of GANs](https://arxiv.org/pdf/1710.10196.pdf)

Trained on CIFAR10 (pggan-cifar10.py) -- requires more training (more gpus)!

![Generator at 4x4](https://github.com/Tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level1.gif)
![Generator at 8x8](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level2.gif)
![Generator at 16x16](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level3.gif)
![Generator at 32x32](https://github.com/tensor46/TensorMONK/blob/develop/models/pggan-cifar10-level4.gif)

## References

### Activation Functions
* [Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf)
* [SWISH: A SELF-GATED ACTIVATION FUNCTION](https://arxiv.org/pdf/1710.05941v1.pdf)

### Classification
* [Deep Neural Decision Forests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)

### Generative Models
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114v10.pdf)
* [Progressive Growing of GANs](https://arxiv.org/pdf/1710.10196.pdf)
* [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)

### Image Recognition Models
* [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
* [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf)
* [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)
* [Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions](https://arxiv.org/pdf/1711.08141.pdf)
* [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
* [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

### Image Segmentation Models
* [AnatomyNet: Deep Learning for Fast and Fully Automated Whole-volume Segmentation of Head and Neck Anatomy](https://arxiv.org/pdf/1808.05238.pdf)
* [ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time](https://arxiv.org/pdf/1805.04554.pdf)
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

### Local Features
* [Learning Discriminative and Transformation Covariant Local Feature Detectors](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Discriminative_and_CVPR_2017_paper.pdf)

### Loss Functions
* [AN EXPLORATION OF SOFTMAX ALTERNATIVES BELONGING TO THE SPHERICAL LOSS FAMILY](https://arxiv.org/pdf/1511.05042.pdf)
* [Analyzing and Improving Representations with the Soft Nearest Neighbor Loss](https://arxiv.org/pdf/1902.01889.pdf)
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)
* [Rethinking Feature Distribution for Loss Functions in Image Classification](https://arxiv.org/pdf/1803.02988.pdf)
* [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/pdf/1706.05721.pdf)

### Object Detection Models
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) (MobileNetV2SSD320 and TinySSD320)

### Regularizations
* [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf)

### Optimizer
* [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/pdf/1907.08610.pdf)
