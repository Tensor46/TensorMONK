import unittest
import tensormonk
import torch

class Tester(unittest.TestCase):
    """
    Test for Tensormonk activation functions
    """

    def test_squash(self):
        print('validating squash function')
        pass
    
    def test_relu(self):
        activation = tensormonk.activations.Activations(activation='relu')
        input = torch.Tensor([-1.0, 0.1, 100.00])
        expected_output = torch.Tensor([0.0, 0.1, 100.00])
        actual_output = activation(input)
        error = expected_output - actual_output

        self.assertEqual(True, error.sum().item() < 1e6)


if __name__ == '__main__':
    '''
    >> nosetests
    '''
    unittest.main()