tensormonk.detection
=======================

Implementation may vary when compared to what is referred, as the intension was
not to replicate but to have the flexibility to utilize concepts across several
papers.

.. contents:: Contents
    :local:

AnchorDetector
--------------------

.. autoclass:: tensormonk.detection.AnchorDetector
    :members:
    :exclude-members: forward

Block
--------------------

.. autoclass:: tensormonk.detection.Block
    :members:
    :exclude-members: forward

Classifier
--------------------

.. autoclass:: tensormonk.detection.Classifier
    :members:
    :exclude-members: forward

CONFIG
--------------------

.. autoclass:: tensormonk.detection.CONFIG
    :members:

FPN Layers
--------------------

All FPN layers use DepthWiseSeparable convolution (with BatchNorm2d and Swish) and FeatureFusion layer.

BiFPNLayer
^^^^^^^^^^^^^^^^^^
.. autoclass:: tensormonk.detection.BiFPNLayer
    :members:
    :exclude-members: forward

FPNLayer
^^^^^^^^^^^^^^^^^^
.. autoclass:: tensormonk.detection.FPNLayer
    :members:
    :exclude-members: forward

NoFPNLayer
^^^^^^^^^^^^^^^^^^
.. autoclass:: tensormonk.detection.NoFPNLayer
    :members:
    :exclude-members: forward

PAFPNLayer
^^^^^^^^^^^^^^^^^^
.. autoclass:: tensormonk.detection.PAFPNLayer
    :members:
    :exclude-members: forward

Responses
--------------------

.. autoclass:: tensormonk.detection.Responses
    :members:

Sample
--------------------

.. autoclass:: tensormonk.detection.Sample
    :members:
