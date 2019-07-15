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
* SeparableConvolution: A 3x3 with groups=channels, followed by 1x1.
* PrimaryCapsule: Primary Capsule for Dynamic routing between capsules
* RoutingCapsule: Routing Capsule for Dynamic routing between capsules
* DetailPooling: Detailed pooling layer
* DoG: Computes difference of two blurred tensors with different gaussian kernels
* DoGBlob: Aggregates several DoG's
* GaussianBlur: Creates Gaussian kernel given sigma and width. n_stds is fixed to 3.
* DoH: Computes determinant of Hessian of BCHW torch.Tensor using the cornor pixels of widthxwidth patch
* HessianBlob: Aggregates determinant of Hessian with width ranging from min_width to max_width (skips every other).
* SSIM: Structural Similarity Index
* SelfAttention: Self-Attention from SAGAN
