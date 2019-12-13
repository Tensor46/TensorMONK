""" TensorMONK's :: unittests :: architectures """

import unittest
import torch
import sys
sys.path.append("../TensorMONK")


class Tester(unittest.TestCase):

    def test_mnas_050(self):
        print("\tcheck -- tensormonk.architectures.MNAS (mnas_050)")
        test = tensormonk.architectures.MNAS(architecture="mnas_050",
                                             predict_imagenet=True)
        test.eval()
        tensor = test.preprocess("./unittests/lion_fish.jpg")
        predicted = torch.argsort(test(tensor).view(-1).softmax(0))
        self.assertEqual(predicted[-1].item(), 396)

    def test_mnas_100(self):
        print("\tcheck -- tensormonk.architectures.MNAS (mnas_100)")
        test = tensormonk.architectures.MNAS(architecture="mnas_100",
                                             predict_imagenet=True)
        test.eval()
        tensor = test.preprocess("./unittests/lion_fish.jpg")
        predicted = torch.argsort(test(tensor).view(-1).softmax(0))
        self.assertEqual(predicted[-1].item(), 396)

    def test_mobilev2(self):
        print("\tcheck -- tensormonk.architectures.MobileNetV2")
        test = tensormonk.architectures.MobileNetV2(predict_imagenet=True)
        test.eval()
        tensor = test.preprocess("./unittests/lion_fish.jpg")
        predicted = torch.argsort(test(tensor).view(-1).softmax(0))
        self.assertEqual(predicted[-1].item(), 396)


if __name__ == '__main__':
    import tensormonk
    unittest.main()
