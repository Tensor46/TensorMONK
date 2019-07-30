### Loss Functions
* [CapsuleLoss](https://arxiv.org/pdf/1710.09829.pdf)
```python
# Example
# Capsule loss -- assuming, the output of routing capsule is of size nx10x32,
# where n is batch size, 10 is number of labels, and 32 is the features from a
# capsule
loss_fn = tensormonk.loss.CapsuleLoss(n_labels=10)
# ex: values
routing_features = torch.rand(4, 10, 32)
targets = torch.Tensor([4, 6, 4, 6]).long()
# usage
loss, (top1, top5) = loss_fn(routing_features, targets)
loss.backward()
```
* Categorical: Cross entropy / [taylor softmax](https://arxiv.org/pdf/1511.05042.pdf) / [additive angular margin loss](https://arxiv.org/pdf/1801.07698.pdf) / [large margin cosine loss](https://arxiv.org/pdf/1801.09414.pdf) / [large-margin Gaussian Mixture](https://arxiv.org/pdf/1803.02988.pdf) / [soft nearest neighbor loss](https://arxiv.org/pdf/1902.01889.pdf)
```python
# Examples
# Enabling hard negative mining with additive angular margin loss
loss_fn = tensormonk.loss.Categorical(
    tensor_size=(1, 64), n_labels=10,
    loss_type="angular_margin", scale=30., margin=0.3,
    add_hard_negative=True, hard_negative_p=0.2)
# usage
embedding = some_network(tensor)
loss, (top1, top5) = loss_fn(embedding, targets)
loss.backward()

# Enabling center and focal loss for a 2 class problem with taylor softmax
loss_fn = tensormonk.loss.Categorical(
    tensor_size=(1, 64), n_labels=2,
    loss_type="taylor_smax", measure="dot",
    add_center=True, center_alpha=0.01, center_scale=0.5
    add_focal=True, focal_alpha=torch.Tensor([1.]*2), focal_gamma=2.)
```
* [DiceLoss / Tversky Loss](https://arxiv.org/pdf/1706.05721.pdf): Segmentation loss function
* MetricLoss: [triplet](https://arxiv.org/pdf/1503.03832.pdf), [angular_triplet](https://arxiv.org/pdf/1708.01682.pdf), [n-pair](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
```python
# Example
from tensormonk.loss import MetricLoss

# Using triplet
loss_fn = MetricLoss(tensor_size=(None, 4), n_labels=10, loss_type="triplet",
                     measure="euclidean", margin=0.5)
# usage
embedding = torch.Tensor([[0.10, 0.60, 0.20, 0.10],
                          [0.15, 0.50, 0.22, 0.11],
                          [0.90, 0.50, 0.96, 0.11],
                          [0.90, 0.10, 0.26, 0.71],
                          [0.85, 0.20, 0.27, 0.78],
                          [0.01, 0.90, 0.91, 0.92],
                          [0.80, 0.45, 0.86, 0.16],
                          [0.92, 0.56, 0.99, 0.06],
                          [0.08, 0.56, 0.16, 0.16]])
targets = torch.Tensor([4, 4, 6, 9, 9, 0, 6, 6, 4]).long()
loss = loss_fn(embedding, targets)
loss.backward()
# triplet sampling is not required - works with any batch with few repeated labels
targets = torch.Tensor([4, 4, 1, 1, 9, 0, 2, 3, 4]).long()
loss = loss_fn(embedding, targets)
loss.backward()
# Using triplet with hard negative mining
loss_fn = MetricLoss(tensor_size=(None, 4), n_labels=10, loss_type="triplet",
                     measure="euclidean", margin=0.5, mining="hard")
# Using n-pair with semi-hard mining
loss_fn = MetricLoss(tensor_size=(None, 4), n_labels=10, loss_type="n_pair",
                     measure="euclidean", margin=0.5, mining="semi")
```
* [MultiBoxLoss](https://arxiv.org/pdf/1512.02325.pdf): Single Shot MultiBox Detector (SSD) loss function
```python
# Example
# Using multi-box loss for input of size 300x300 (SSD300)
from tensormonk.utils import SSDUtils
from tensormonk.loss import MultiBoxLoss

translator = SSDUtils.Translator(model="SSD300", var1=.1, var2=.2,
                                 encode_iou_threshold=0.5)
loss_fn = MultiBoxLoss(translator, neg_pos_ratio=3., alpha=1.)
# ex values from network
gcxcywh_boxes = torch.rand(2, 8732, 4)
predictions = torch.rand(2, 8732, 3)
# ex values (ground truth)
target_boxes = (torch.Tensor([[0.1, 0.1, 0.6, 0.9]]),
                torch.Tensor([[0.6, 0.8, 0.6, 0.9], [0.2, 0.3, 0.4, 0.6]]))
targets = (torch.Tensor([0]).long(), torch.Tensor([0, 2]).long())

loss = test(gcxcywh_boxes, predictions, target_boxes, targets)
loss.backward()
```
* TripletLoss: With soft and hard negative mining
