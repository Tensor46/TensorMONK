### Loss Functions
* [CapsuleLoss](https://arxiv.org/pdf/1710.09829.pdf)
* Categorical: Cross entropy / [taylor softmax](https://arxiv.org/pdf/1511.05042.pdf) / [additive angular margin loss](https://arxiv.org/pdf/1801.07698.pdf)[large margin cosine loss](https://arxiv.org/pdf/1801.09414.pdf) / [large-margin Gaussian Mixture](https://arxiv.org/pdf/1803.02988.pdf) / [soft nearest neighbor loss](https://arxiv.org/pdf/1902.01889.pdf)
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
* [MultiBoxLoss](https://arxiv.org/pdf/1512.02325.pdf): Single Shot MultiBox Detector (SSD) loss function
```python
# Example
# Using multi-box loss for SSD of input size 320x320
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
