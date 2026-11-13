# import torch
#
# from dl_lib.vision.conv import Conv1d, Conv2d
#
#
# class TestConv1dInit(unittest.TestCase):
#     def test_when_constructed_then_weight_and_bias_have_torch_shapes(self):
#         # Arrange
#         in_channels, out_channels, kernel_size = 3, 4, 5
#         expected_weight_shape = (out_channels, in_channels, kernel_size)
#         expected_bias_shape = (out_channels,)
#
#         # Act
#         conv = Conv1d(in_channels, out_channels, kernel_size)
#
#         # Assert
#         self.assertEqual(tuple(conv.weight.shape), expected_weight_shape)
#         self.assertEqual(tuple(conv.bias.shape), expected_bias_shape)
#
#     def test_when_constructed_then_weight_and_bias_require_grad(self):
#         # Arrange & Act
#         conv = Conv1d(in_channels=2, out_channels=2, kernel_size=3)
#
#         # Assert
#         self.assertTrue(conv.weight.requires_grad)
#         self.assertTrue(conv.bias.requires_grad)
#
#     def test_when_channels_and_stride_and_padding_not_given_then_default_to_one_and_zero(
#         self,
#     ):
#         # Arrange
#         expected_in_channels, expected_out_channels = 1, 1
#         expected_stride, expected_padding = 1, 0
#
#         # Act
#         conv = Conv1d(kernel_size=3)
#
#         # Assert
#         self.assertEqual(conv.in_channels, expected_in_channels)
#         self.assertEqual(conv.out_channels, expected_out_channels)
#         self.assertEqual(conv.stride, expected_stride)
#         self.assertEqual(conv.padding, expected_padding)
#
#
# class TestConv1dForward(unittest.TestCase):
#     def test_when_input_is_unbatched_then_output_is_unbatched(self):
#         # Arrange
#         conv = Conv1d(in_channels=1, out_channels=1, kernel_size=3)
#         x = torch.randn(1, 10)
#
#         # Act
#         actual = conv(x)
#
#         # Assert
#         self.assertEqual(actual.dim(), 2)
#
#     def test_when_input_is_batched_then_output_keeps_the_batch_dimension(self):
#         # Arrange
#         batch_size = 4
#         conv = Conv1d(in_channels=1, out_channels=1, kernel_size=3)
#         x = torch.randn(batch_size, 1, 10)
#
#         # Act
#         actual = conv(x)
#
#         # Assert
#         self.assertEqual(actual.shape[0], batch_size)
#
#     def test_when_stride_and_padding_given_then_output_length_matches_the_formula(self):
#         # Arrange
#         length, kernel_size, stride, padding = 10, 3, 2, 1
#         expected_length = (length + 2 * padding - kernel_size) // stride + 1
#         conv = Conv1d(
#             in_channels=1,
#             out_channels=1,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         x = torch.randn(1, length)
#
#         # Act
#         actual = conv(x)
#
#         # Assert
#         self.assertEqual(actual.shape[-1], expected_length)
#
#     def test_when_single_channel_unbatched_then_matches_torch_nn_conv1d(self):
#         # Arrange
#         torch.manual_seed(0)
#         kernel_size = 3
#         x = torch.randn(1, 12)
#         ours = Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size)
#         reference = torch.nn.Conv1d(
#             in_channels=1, out_channels=1, kernel_size=kernel_size
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_multiple_channels_and_batched_then_matches_torch_nn_conv1d(self):
#         # Arrange
#         torch.manual_seed(1)
#         in_channels, out_channels, kernel_size, batch_size, length = 3, 4, 5, 2, 20
#         x = torch.randn(batch_size, in_channels, length)
#         ours = Conv1d(in_channels, out_channels, kernel_size)
#         reference = torch.nn.Conv1d(in_channels, out_channels, kernel_size)
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_stride_given_then_matches_torch_nn_conv1d(self):
#         # Arrange
#         torch.manual_seed(2)
#         in_channels, out_channels, kernel_size, stride, length = 2, 3, 3, 2, 17
#         x = torch.randn(2, in_channels, length)
#         ours = Conv1d(in_channels, out_channels, kernel_size, stride=stride)
#         reference = torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size, stride=stride
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_padding_given_then_matches_torch_nn_conv1d(self):
#         # Arrange
#         torch.manual_seed(3)
#         in_channels, out_channels, kernel_size, padding, length = 2, 3, 3, 2, 17
#         x = torch.randn(2, in_channels, length)
#         ours = Conv1d(in_channels, out_channels, kernel_size, padding=padding)
#         reference = torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size, padding=padding
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_stride_and_padding_both_given_then_matches_torch_nn_conv1d(self):
#         # Arrange
#         torch.manual_seed(4)
#         in_channels, out_channels, kernel_size, stride, padding, length = (
#             3,
#             2,
#             4,
#             3,
#             2,
#             25,
#         )
#         x = torch.randn(3, in_channels, length)
#         ours = Conv1d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         reference = torch.nn.Conv1d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#
# class TestConv1dParameters(unittest.TestCase):
#     def test_when_called_then_returns_weight_and_bias(self):
#         # Arrange
#         conv = Conv1d(in_channels=2, out_channels=3, kernel_size=3)
#         expected = [conv.weight, conv.bias]
#
#         # Act
#         actual = conv.parameters()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_then_the_returned_weight_and_bias_are_the_same_objects(self):
#         # Arrange
#         conv = Conv1d(in_channels=2, out_channels=3, kernel_size=3)
#
#         # Act
#         actual = conv.parameters()
#
#         # Assert
#         self.assertIs(actual[0], conv.weight)
#         self.assertIs(actual[1], conv.bias)
#
#
# class TestConv2dInit(unittest.TestCase):
#     def test_when_constructed_then_weight_and_bias_have_torch_shapes(self):
#         # Arrange
#         in_channels, out_channels, kernel_size = 3, 8, 5
#         expected_weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
#         expected_bias_shape = (out_channels,)
#
#         # Act
#         conv = Conv2d(in_channels, out_channels, kernel_size)
#
#         # Assert
#         self.assertEqual(tuple(conv.weight.shape), expected_weight_shape)
#         self.assertEqual(tuple(conv.bias.shape), expected_bias_shape)
#
#     def test_when_stride_and_padding_not_given_then_default_to_one_and_zero(self):
#         # Arrange
#         expected_stride, expected_padding = 1, 0
#
#         # Act
#         conv = Conv2d(in_channels=3, out_channels=8, kernel_size=5)
#
#         # Assert
#         self.assertEqual(conv.stride, expected_stride)
#         self.assertEqual(conv.padding, expected_padding)
#
#
# class TestConv2dForward(unittest.TestCase):
#     def test_when_input_is_unbatched_then_output_is_unbatched(self):
#         # Arrange
#         conv = Conv2d(in_channels=3, out_channels=2, kernel_size=3)
#         x = torch.randn(3, 10, 10)
#
#         # Act
#         actual = conv(x)
#
#         # Assert
#         self.assertEqual(actual.dim(), 3)
#
#     def test_when_stride_and_padding_given_then_output_shape_matches_the_formula(self):
#         # Arrange
#         height, width, kernel_size, stride, padding = 16, 16, 3, 2, 1
#         expected_height = (height + 2 * padding - kernel_size) // stride + 1
#         expected_width = (width + 2 * padding - kernel_size) // stride + 1
#         conv = Conv2d(
#             in_channels=3,
#             out_channels=4,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         x = torch.randn(2, 3, height, width)
#
#         # Act
#         actual = conv(x)
#
#         # Assert
#         self.assertEqual(actual.shape[-2], expected_height)
#         self.assertEqual(actual.shape[-1], expected_width)
#
#     def test_when_given_same_weights_then_matches_torch_nn_conv2d(self):
#         # Arrange
#         torch.manual_seed(5)
#         in_channels, out_channels, kernel_size = 3, 4, 3
#         x = torch.randn(2, in_channels, 12, 12)
#         ours = Conv2d(in_channels, out_channels, kernel_size)
#         reference = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_stride_and_padding_given_then_matches_torch_nn_conv2d(self):
#         # Arrange
#         torch.manual_seed(6)
#         in_channels, out_channels, kernel_size, stride, padding = 3, 4, 5, 2, 2
#         x = torch.randn(2, in_channels, 20, 20)
#         ours = Conv2d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         reference = torch.nn.Conv2d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_given_alexnet_conv1_config_then_matches_torch_nn_conv2d(self):
#         # Arrange
#         torch.manual_seed(7)
#         in_channels, out_channels, kernel_size, stride, padding = 3, 96, 11, 4, 2
#         x = torch.randn(2, in_channels, 64, 64)
#         ours = Conv2d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         reference = torch.nn.Conv2d(
#             in_channels, out_channels, kernel_size, stride=stride, padding=padding
#         )
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             assert reference.bias is not None
#             ours.bias.copy_(reference.bias)
#
#         # Act
#         actual = ours(x)
#         expected = reference(x)
#
#         # Assert
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#
# class TestConv2dParameters(unittest.TestCase):
#     def test_when_called_then_returns_weight_and_bias(self):
#         # Arrange
#         conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3)
#         expected = [conv.weight, conv.bias]
#
#         # Act
#         actual = conv.parameters()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_then_the_returned_weight_and_bias_are_the_same_objects(self):
#         # Arrange
#         conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3)
#
#         # Act
#         actual = conv.parameters()
#
#         # Assert
#         self.assertIs(actual[0], conv.weight)
#         self.assertIs(actual[1], conv.bias)
