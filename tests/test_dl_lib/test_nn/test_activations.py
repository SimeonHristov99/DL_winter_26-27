# import unittest

# import torch

# from dl_lib.nn.activations import LeakyReLU, ReLU, Sigmoid, Tanh


# class TestSigmoidCall(unittest.TestCase):
#     def test_when_input_is_zero_then_returns_one_half(self):
#         # Arrange
#         sigmoid = Sigmoid()
#         x = torch.tensor([0.0])
#         expected = torch.tensor([0.5])

#         # Act
#         actual = sigmoid(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))

#     def test_when_called_then_matches_torch_sigmoid(self):
#         # Arrange
#         sigmoid = Sigmoid()
#         x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
#         expected = torch.sigmoid(x)

#         # Act
#         actual = sigmoid(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))


# class TestTanhCall(unittest.TestCase):
#     def test_when_input_is_zero_then_returns_zero(self):
#         # Arrange
#         tanh = Tanh()
#         x = torch.tensor([0.0])
#         expected = torch.tensor([0.0])

#         # Act
#         actual = tanh(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))

#     def test_when_called_then_matches_torch_tanh(self):
#         # Arrange
#         tanh = Tanh()
#         x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
#         expected = torch.tanh(x)

#         # Act
#         actual = tanh(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))


# class TestReLUCall(unittest.TestCase):
#     def test_when_input_has_negative_values_then_zeroes_them_out(self):
#         # Arrange
#         relu = ReLU()
#         x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
#         expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])

#         # Act
#         actual = relu(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, expected))

#     def test_when_input_is_all_non_negative_then_leaves_it_unchanged(self):
#         # Arrange
#         relu = ReLU()
#         x = torch.tensor([0.0, 1.0, 2.0])
#         expected = torch.tensor([0.0, 1.0, 2.0])

#         # Act
#         actual = relu(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, expected))


# class TestLeakyReLUInit(unittest.TestCase):
#     def test_when_negative_slope_not_given_then_defaults_to_one_hundredth(self):
#         # Arrange
#         expected = 1e-2

#         # Act
#         actual = LeakyReLU().negative_slope

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_negative_slope_given_then_stores_it(self):
#         # Arrange
#         expected = 0.3

#         # Act
#         actual = LeakyReLU(negative_slope=0.3).negative_slope

#         # Assert
#         self.assertEqual(actual, expected)


# class TestLeakyReLUCall(unittest.TestCase):
#     def test_when_input_has_negative_values_then_scales_them_by_negative_slope(self):
#         # Arrange
#         leaky_relu = LeakyReLU(negative_slope=0.1)
#         x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
#         expected = torch.tensor([-0.2, -0.1, 0.0, 1.0, 2.0])

#         # Act
#         actual = leaky_relu(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))

#     def test_when_negative_slope_not_given_then_scales_negatives_by_default_value(
#             self):
#         # Arrange
#         leaky_relu = LeakyReLU()
#         x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
#         expected = torch.tensor([-0.02, -0.01, 0.0, 1.0, 2.0])

#         # Act
#         actual = leaky_relu(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))
