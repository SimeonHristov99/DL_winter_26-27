import unittest

import torch

from dl_lib.optim.optimizer import Optimizer


class _Zero(Optimizer):
    """Minimal concrete `Optimizer` that never updates a parameter, used only to test the ABC."""

    def step(self) -> None:
        pass


class TestInit(unittest.TestCase):
    def test_when_constructed_then_stores_the_given_parameters_and_learning_rate(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_params = [p]
        expected_lr = 0.1

        # Act
        optimizer = _Zero([p], lr=0.1)

        # Assert
        self.assertEqual(optimizer.params, expected_params)
        self.assertEqual(optimizer.lr, expected_lr)

    def test_when_constructed_with_multiple_parameters_then_stores_them_in_order(self):
        # Arrange
        p1 = torch.tensor([1.0])
        p2 = torch.tensor([2.0])
        expected_params = [p1, p2]
        expected_lr = 0.5

        # Act
        optimizer = _Zero([p1, p2], lr=0.5)

        # Assert
        self.assertEqual(optimizer.params, expected_params)
        self.assertEqual(optimizer.lr, expected_lr)


class TestZeroGrad(unittest.TestCase):
    def test_when_parameter_has_a_gradient_then_zeroes_it_in_place(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        p.grad = torch.tensor([3.0, 4.0])
        optimizer = _Zero([p], lr=0.1)
        expected = torch.zeros_like(p)

        # Act
        optimizer.zero_grad()

        # Assert
        self.assertTrue(torch.equal(p.grad, expected))

    def test_when_parameter_has_no_gradient_then_leaves_it_as_none(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        optimizer = _Zero([p], lr=0.1)

        # Act
        optimizer.zero_grad()

        # Assert
        self.assertIsNone(p.grad)
