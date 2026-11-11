# import unittest

# import torch

# from dl_lib.nn import module
# from dl_lib.nn.dropout import Dropout


# class TestInit(unittest.TestCase):
#     def test_when_constructed_without_p_then_defaults_to_one_half(self):
#         # Arrange
#         expected = 0.5
#         expected_class = module.Module

#         # Act
#         dropout = Dropout()
#         actual_p = dropout.p
#         actual_training = dropout.training

#         # Assert
#         self.assertEqual(actual_p, expected)
#         self.assertIsInstance(dropout, expected_class)
#         self.assertTrue(actual_training)

#     def test_when_p_given_then_stores_it(self):
#         # Arrange
#         expected = 0.3

#         # Act
#         actual = Dropout(p=0.3).p

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_p_is_negative_then_raises_value_error(self):
#         # Arrange
#         p = -0.1

#         # Act / Assert
#         with self.assertRaises(ValueError):
#             Dropout(p=p)

#     def test_when_p_is_positive_then_raises_value_error(self):
#         # Arrange
#         p = 1.1

#         # Act / Assert
#         with self.assertRaises(ValueError):
#             Dropout(p=p)


# class TestForward(unittest.TestCase):
#     def test_when_evaluating_then_returns_input_unchanged(self):
#         # Arrange
#         dropout = Dropout(p=0.5)
#         dropout.eval()
#         x = torch.tensor([1.0, 2.0, 3.0])

#         # Act
#         actual = dropout(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, x))

#     def test_when_p_is_zero_then_returns_input_unchanged_even_in_training(self):
#         # Arrange
#         dropout = Dropout(p=0.0)
#         x = torch.tensor([1.0, 2.0, 3.0])

#         # Act
#         actual = dropout(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, x))

#     def test_when_training_then_every_element_is_zero_or_scaled(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.5)
#         x = torch.ones(1000)
#         survivor_value = torch.tensor(2.0)

#         # Act
#         actual = dropout(x)
#         is_zero_or_survivor = (actual == 0.0) | torch.isclose(actual, survivor_value)

#         # Assert
#         self.assertTrue(torch.all(is_zero_or_survivor))

#     def test_when_training_then_drops_approximately_p_fraction_of_elements(self):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.3)
#         x = torch.ones(10000)
#         expected_fraction = 0.3

#         # Act
#         actual = dropout(x)

#         # Assert
#         fraction_zeroed = (actual == 0.0).float().mean().item()
#         self.assertAlmostEqual(fraction_zeroed, expected_fraction, delta=0.02)

#     def test_when_input_is_batched_then_preserves_shape_and_scales_elements(self):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.3)
#         x = torch.arange(1, 4097, dtype=torch.float32).reshape(32, 128)
#         expected_survivors = x / (1.0 - dropout.p)

#         # Act
#         actual = dropout(x)

#         # Assert
#         self.assertEqual(actual.shape, x.shape)
#         is_zero_or_survivor = (actual == 0.0) | torch.isclose(
#             actual, expected_survivors
#         )
#         self.assertTrue(torch.all(is_zero_or_survivor).item())

#     def test_when_input_is_batched_then_drops_approximately_p_fraction(self):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.3)
#         x = torch.ones(100, 100)

#         # Act
#         actual = dropout(x)

#         # Assert
#         fraction_zeroed = (actual == 0.0).float().mean().item()
#         self.assertAlmostEqual(fraction_zeroed, dropout.p, delta=0.02)

#     def test_when_input_is_batched_then_rows_do_not_share_one_dropout_mask(self):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.5)
#         x = torch.ones(32, 128)

#         # Act
#         actual = dropout(x)

#         # Assert
#         dropped = actual == 0.0
#         all_rows_share_first_mask = torch.all(dropped == dropped[0]).item()
#         self.assertFalse(all_rows_share_first_mask)

#     def test_when_training_then_backprop_uses_forward_mask_and_scaling(self):
#         # Arrange
#         torch.manual_seed(0)
#         dropout = Dropout(p=0.5)
#         x = torch.ones(1000, requires_grad=True)
#         incoming_gradient = torch.linspace(0.1, 2.0, steps=x.numel())

#         # Act
#         output = dropout(x)
#         output.backward(incoming_gradient)

#         # Assert
#         self.assertIsNotNone(x.grad)
#         survived = output.detach() != 0.0
#         self.assertTrue(survived.any().item())
#         self.assertTrue((~survived).any().item())

#         expected_gradient = incoming_gradient * survived / (1.0 - dropout.p)
#         torch.testing.assert_close(x.grad, expected_gradient)

#     def test_when_evaluating_then_backprop_passes_gradient_unchanged(self):
#         # Arrange
#         dropout = Dropout(p=0.5)
#         dropout.eval()
#         x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#         incoming_gradient = torch.tensor([0.5, -2.0, 3.0])

#         # Act
#         dropout(x).backward(incoming_gradient)

#         # Assert
#         torch.testing.assert_close(x.grad, incoming_gradient)

#     def test_when_p_is_zero_then_backprop_passes_gradient_unchanged(self):
#         # Arrange
#         dropout = Dropout(p=0.0)
#         x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#         incoming_gradient = torch.tensor([0.5, -2.0, 3.0])

#         # Act
#         dropout(x).backward(incoming_gradient)

#         # Assert
#         torch.testing.assert_close(x.grad, incoming_gradient)

#     def test_when_p_is_one_then_output_and_input_gradient_are_zero(self):
#         # Arrange
#         dropout = Dropout(p=1.0)
#         x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

#         # Act
#         output = dropout(x)
#         output.sum().backward()

#         # Assert
#         torch.testing.assert_close(output, torch.zeros_like(x))
#         self.assertIsNotNone(x.grad)
#         torch.testing.assert_close(x.grad, torch.zeros_like(x))
