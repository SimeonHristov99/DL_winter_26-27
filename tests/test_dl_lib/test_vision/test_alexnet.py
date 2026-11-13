# === SOLUTION SET-ASIDE: this test file is commented out. ===
# Uncomment the lines below (strip the leading '# ') to re-enable it and drive
# your reconstruction RED -> GREEN. The reference implementation lives under
# solutions/dl_lib/<topic>/. Original content follows, one '# ' prefix deep:
# import unittest
#
# import torch
#
# from dl_lib.vision.alexnet import AlexNet
#
#
# class TestInit(unittest.TestCase):
#     def test_when_constructed_with_three_input_channels_then_features_output_is_256_channels(
#         self,
#     ):
#         # Arrange
#         model = AlexNet(in_channels=3)
#         x = torch.randn(2, 3, 64, 64)
#         expected = torch.Size([2, 256, 1, 1])
#
#         # Act
#         actual = model.features(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_with_one_input_channel_then_only_the_first_conv_layer_changes(
#         self,
#     ):
#         # Arrange
#         model = AlexNet(in_channels=1)
#         x = torch.randn(2, 1, 64, 64)
#         expected = torch.Size([2, 256, 1, 1])
#
#         # Act
#         actual = model.features(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_then_defaults_to_training_mode(self):
#         # Arrange
#         model = AlexNet()
#
#         # Act
#         actual = model.training
#
#         # Assert
#         self.assertTrue(actual)
#
#
# class TestTo(unittest.TestCase):
#     def test_when_called_with_cpu_then_every_parameter_stays_on_cpu(self):
#         # Arrange
#         model = AlexNet()
#         expected = torch.device("cpu")
#
#         # Act
#         model.to(expected)
#
#         # Assert
#         for param in model.parameters():
#             self.assertEqual(param.device, expected)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = AlexNet()
#
#         # Act
#         actual = model.to(torch.device("cpu"))
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestForward(unittest.TestCase):
#     def test_when_called_then_output_shape_matches_batch_size_and_num_classes(self):
#         # Arrange
#         model = AlexNet(num_classes=10)
#         x = torch.randn(4, 3, 64, 64)
#         expected = torch.Size([4, 10])
#
#         # Act
#         actual = model(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_then_output_is_finite(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = AlexNet()
#         x = torch.randn(4, 3, 64, 64)
#
#         # Act
#         actual = model(x)
#
#         # Assert
#         self.assertFalse(torch.isnan(actual).any())
#         self.assertTrue(torch.isfinite(actual).all())
#
#     def test_when_evaluating_then_two_forward_passes_of_the_same_input_are_identical(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         model = AlexNet()
#         model.eval()
#         x = torch.randn(2, 3, 64, 64)
#
#         # Act
#         first = model(x)
#         second = model(x)
#
#         # Assert
#         self.assertTrue(torch.equal(first, second))
#
#     def test_when_training_then_two_forward_passes_of_the_same_input_generally_differ(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         model = AlexNet()
#         x = torch.randn(2, 3, 64, 64)
#
#         # Act
#         first = model(x)
#         second = model(x)
#
#         # Assert
#         self.assertFalse(torch.equal(first, second))
#
#     def test_when_backward_is_called_then_every_parameter_gets_a_finite_nonzero_gradient(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         model = AlexNet(num_classes=10)
#         x = torch.randn(2, 3, 64, 64)
#         y = torch.tensor([0, 1])
#         loss_fn = torch.nn.CrossEntropyLoss()
#
#         # Act
#         loss = loss_fn(model(x), y)
#         loss.backward()
#
#         # Assert
#         total_grad_magnitude = 0.0
#         for param in model.parameters():
#             self.assertIsNotNone(param.grad)
#             self.assertFalse(torch.isnan(param.grad).any())
#             total_grad_magnitude += param.grad.abs().sum().item()
#         self.assertGreater(total_grad_magnitude, 0.0)
#
#     def test_when_trained_for_a_few_steps_then_loss_decreases(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = AlexNet(num_classes=2)
#         x = torch.randn(8, 3, 64, 64)
#         y = torch.randint(0, 2, (8,))
#         loss_fn = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#
#         initial_loss = loss_fn(model(x), y).item()
#
#         # Act
#         for _ in range(5):
#             optimizer.zero_grad()
#             loss = loss_fn(model(x), y)
#             loss.backward()
#             optimizer.step()
#         final_loss = loss_fn(model(x), y).item()
#
#         # Assert
#         self.assertFalse(torch.isnan(torch.tensor(final_loss)))
#         self.assertLess(final_loss, initial_loss)
#
#
# class TestTrain(unittest.TestCase):
#     def test_when_called_after_eval_then_sets_features_and_classifier_to_training_mode(
#         self,
#     ):
#         # Arrange
#         model = AlexNet()
#         model.eval()
#
#         # Act
#         model.train()
#
#         # Assert
#         self.assertTrue(model.training)
#         self.assertTrue(model.features.training)
#         self.assertTrue(model.classifier.training)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = AlexNet()
#
#         # Act
#         actual = model.train()
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestEval(unittest.TestCase):
#     def test_when_called_then_sets_features_and_classifier_to_evaluation_mode(self):
#         # Arrange
#         model = AlexNet()
#
#         # Act
#         model.eval()
#
#         # Assert
#         self.assertFalse(model.training)
#         self.assertFalse(model.features.training)
#         self.assertFalse(model.classifier.training)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = AlexNet()
#
#         # Act
#         actual = model.eval()
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestParameters(unittest.TestCase):
#     def test_when_called_then_returns_every_parameter_from_features_and_classifier(
#         self,
#     ):
#         # Arrange
#         model = AlexNet()
#         expected = len(model.features.parameters()) + len(model.classifier.parameters())
#
#         # Act
#         actual = len(model.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#         self.assertGreater(actual, 0)
#
#     def test_when_called_then_returns_features_then_classifier_parameters_in_order(
#         self,
#     ):
#         # Arrange
#         model = AlexNet()
#         expected = model.features.parameters() + model.classifier.parameters()
#
#         # Act
#         actual = model.parameters()
#
#         # Assert
#         self.assertEqual(len(actual), len(expected))
#         for actual_param, expected_param in zip(actual, expected):
#             self.assertIs(actual_param, expected_param)
