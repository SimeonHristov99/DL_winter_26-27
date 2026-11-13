# === SOLUTION SET-ASIDE: this test file is commented out. ===
# Uncomment the lines below (strip the leading '# ') to re-enable it and drive
# your reconstruction RED -> GREEN. The reference implementation lives under
# solutions/dl_lib/<topic>/. Original content follows, one '# ' prefix deep:
# import unittest
#
# import torch
#
# from dl_lib.vision.pooling import MaxPool2d
#
#
# class TestMaxPool2dInit(unittest.TestCase):
#     def test_when_stride_not_given_then_defaults_to_kernel_size(self):
#         # Arrange
#         kernel_size = 3
#         expected_stride = kernel_size
#
#         # Act
#         pool = MaxPool2d(kernel_size)
#
#         # Assert
#         self.assertEqual(pool.stride, expected_stride)
#
#     def test_when_padding_not_given_then_defaults_to_zero(self):
#         # Arrange
#         expected_padding = 0
#
#         # Act
#         pool = MaxPool2d(kernel_size=3)
#
#         # Assert
#         self.assertEqual(pool.padding, expected_padding)
#
#
# class TestMaxPool2dForward(unittest.TestCase):
#     def test_when_input_is_unbatched_then_output_is_unbatched(self):
#         # Arrange
#         pool = MaxPool2d(kernel_size=2)
#         x = torch.randn(3, 8, 8)
#
#         # Act
#         actual = pool(x)
#
#         # Assert
#         self.assertEqual(actual.dim(), 3)
#
#     def test_when_input_is_batched_then_output_keeps_the_batch_dimension(self):
#         # Arrange
#         batch_size = 4
#         pool = MaxPool2d(kernel_size=2)
#         x = torch.randn(batch_size, 3, 8, 8)
#
#         # Act
#         actual = pool(x)
#
#         # Assert
#         self.assertEqual(actual.shape[0], batch_size)
#
#     def test_when_stride_and_padding_given_then_output_shape_matches_the_formula(self):
#         # Arrange
#         height, width, kernel_size, stride, padding = 15, 15, 3, 2, 0
#         expected_height = (height + 2 * padding - kernel_size) // stride + 1
#         expected_width = (width + 2 * padding - kernel_size) // stride + 1
#         pool = MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
#         x = torch.randn(2, 3, height, width)
#
#         # Act
#         actual = pool(x)
#
#         # Assert
#         self.assertEqual(actual.shape[-2], expected_height)
#         self.assertEqual(actual.shape[-1], expected_width)
#
#     def test_when_called_then_keeps_the_maximum_value_in_each_window(self):
#         # Arrange
#         pool = MaxPool2d(kernel_size=2, stride=2)
#         x = torch.tensor(
#             [
#                 [
#                     [1.0, 2.0, 5.0, 6.0],
#                     [3.0, 4.0, 7.0, 8.0],
#                     [9.0, 10.0, 13.0, 14.0],
#                     [11.0, 12.0, 15.0, 16.0],
#                 ]
#             ]
#         )
#         expected = torch.tensor([[[4.0, 8.0], [12.0, 16.0]]])
#
#         # Act
#         actual = pool(x)
#
#         # Assert
#         self.assertTrue(torch.equal(actual, expected))
#
#     def test_when_given_same_input_then_matches_torch_nn_maxpool2d(self):
#         # Arrange
#         torch.manual_seed(0)
#         kernel_size, stride = 2, 2
#         x = torch.randn(2, 3, 16, 16)
#         ours = MaxPool2d(kernel_size=kernel_size, stride=stride)
#         reference = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_stride_given_then_matches_torch_nn_maxpool2d(self):
#         # Arrange
#         torch.manual_seed(1)
#         kernel_size, stride = 3, 1
#         x = torch.randn(2, 3, 12, 12)
#         ours = MaxPool2d(kernel_size=kernel_size, stride=stride)
#         reference = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_padding_given_then_matches_torch_nn_maxpool2d(self):
#         # Arrange
#         torch.manual_seed(3)
#         kernel_size, stride, padding = 3, 2, 1
#         x = torch.randn(2, 3, 15, 15)
#         ours = MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
#         reference = torch.nn.MaxPool2d(
#             kernel_size=kernel_size, stride=stride, padding=padding
#         )
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_given_alexnet_pool_config_then_matches_torch_nn_maxpool2d(self):
#         # Arrange
#         torch.manual_seed(2)
#         kernel_size, stride = 3, 2
#         x = torch.randn(2, 96, 15, 15)
#         ours = MaxPool2d(kernel_size=kernel_size, stride=stride)
#         reference = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
