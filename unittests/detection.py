""" TensorMONK's :: unittests :: detection """

import unittest
import torch
import sys
sys.path.append("../TensorMONK")


class Tester(unittest.TestCase):

    def test_utils_pixel_norm01(self):
        pixels = torch.Tensor([[40., 60., 40., 60.]])
        w, h = 60, 60
        norm01 = torch.Tensor([[40./w, 60./h, 40./w, 60./h]])

        print("\tcheck -- tensormonk.detection.ObjectUtils.pixel_to_norm01")
        output = ObjectUtils.pixel_to_norm01(pixels, w, h)
        self.assertTrue((output == norm01).all().item())

        print("\tcheck -- tensormonk.detection.ObjectUtils.norm01_to_pixel")
        output = ObjectUtils.norm01_to_pixel(norm01, w, h)
        self.assertTrue((output == pixels).all().item())

    def test_utils_ltrb_cxcywh(self):
        ltrb = torch.Tensor([[40, 60, 60, 90]])
        cxcywh = torch.Tensor([[50, 75, 20, 30]])

        print("\tcheck -- tensormonk.detection.ObjectUtils.ltrb_to_cxcywh")
        output = ObjectUtils.ltrb_to_cxcywh(ltrb)
        self.assertTrue((output == cxcywh).all().item())

        print("\tcheck -- tensormonk.detection.ObjectUtils.cxcywh_to_ltrb")
        output = ObjectUtils.cxcywh_to_ltrb(cxcywh)
        self.assertTrue((output == ltrb).all().item())

    def test_utils_compute_intersection(self):
        ltrb1 = torch.Tensor([[40, 60, 60, 90]])
        ltrb2 = torch.Tensor([[56, 76, 66, 96]])
        intersection = (60. - 56) * (90 - 76)

        print("\tcheck -- tensormonk.detection.ObjectUtils."
              "compute_intersection")
        output = ObjectUtils.compute_intersection(ltrb1, ltrb2)
        self.assertEqual(output.squeeze().item(), intersection)

    def test_utils_compute_area(self):
        ltrb = torch.Tensor([[40, 60, 60, 90]])
        intersection = 20 * 30

        print("\tcheck -- tensormonk.detection.ObjectUtils.compute_area")
        output = ObjectUtils.compute_area(ltrb)
        self.assertEqual(output.squeeze().item(), intersection)

    def test_utils_compute_iou(self):
        ltrb1 = torch.Tensor([[40, 60, 60, 90]])
        ltrb2 = torch.Tensor([[56, 76, 66, 96]])
        intersection = (60. - 56) * (90 - 76)
        expected = intersection / (20 * 30 + 10 * 20 - intersection)

        print("\tcheck -- tensormonk.detection.ObjectUtils.compute_iou")
        output = ObjectUtils.compute_iou(ltrb1, ltrb2)
        self.assertEqual(round(output.squeeze().item(), 8), round(expected, 8))
        output = ObjectUtils.compute_iou(ltrb1, ltrb1)
        self.assertEqual(output.squeeze().item(), 1.0)

    def test_utils_compute_iof(self):
        ltrb1 = torch.Tensor([[40, 60, 60, 90]])
        ltrb2 = torch.Tensor([[56, 76, 66, 96]])
        intersection = (60. - 56) * (90 - 76)
        expected = intersection / (20 * 30 + 1e-15)

        print("\tcheck -- tensormonk.detection.ObjectUtils.compute_iof")
        output = ObjectUtils.compute_iof(ltrb1, ltrb2)
        self.assertEqual(round(output.squeeze().item(), 8), round(expected, 8))
        output = ObjectUtils.compute_iof(ltrb1, ltrb1)
        self.assertEqual(output.squeeze().item(), 1.0)


if __name__ == '__main__':
    from tensormonk.detection import ObjectUtils
    unittest.main()
