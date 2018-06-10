python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.1 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.01 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
python ImageNet.py -A residual50 -B 32 -E 30 --optimizer sgd --learningRate 0.001 --gpus 1 --cpus 6 --trainDataPath ./data/ImageNet/train --testDataPath ./data/ImageNet/validation
