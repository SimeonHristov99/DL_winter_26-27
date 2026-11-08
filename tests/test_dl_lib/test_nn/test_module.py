# import unittest

# import torch

# from dl_lib.nn.module import Module


# class _DoubleModule(Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * 2.0


# class _NegateModule(Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return -x


# class TestCall(unittest.TestCase):
#     def test_when_called_then_delegates_to_forward(self):
#         # Arrange
#         module = _DoubleModule()
#         x = torch.tensor([1.0, 2.0, 3.0])
#         expected = torch.tensor([2.0, 4.0, 6.0])

#         # Act
#         actual = module(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, expected))

#     def test_when_subclass_overrides_forward_differently_then_call_uses_that_implementation(
#             self):
#         # Arrange
#         module = _NegateModule()
#         x = torch.tensor([1.0, -2.0, 3.0])
#         expected = torch.tensor([-1.0, 2.0, -3.0])

#         # Act
#         actual = module(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, expected))


# class TestParameters(unittest.TestCase):
#     def test_when_called_then_returns_empty_list_by_default(self):
#         # Arrange
#         module = _DoubleModule()
#         expected = []

#         # Act
#         actual = module.parameters()

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_called_on_a_different_subclass_then_still_returns_empty_list(
#             self):
#         # Arrange
#         module = _NegateModule()
#         expected = []

#         # Act
#         actual = module.parameters()

#         # Assert
#         self.assertEqual(actual, expected)


# class TestTrain(unittest.TestCase):
#     def test_when_called_then_sets_training_to_true(self):
#         # Arrange
#         module = _DoubleModule()
#         module.training = False

#         # Act
#         module.train()

#         # Assert
#         self.assertTrue(module.training)

#     def test_when_called_then_returns_self(self):
#         # Arrange
#         module = _DoubleModule()

#         # Act
#         actual = module.train()

#         # Assert
#         self.assertIs(actual, module)


# class TestEval(unittest.TestCase):
#     def test_when_called_then_sets_training_to_false(self):
#         # Arrange
#         module = _DoubleModule()

#         # Act
#         module.eval()

#         # Assert
#         self.assertFalse(module.training)

#     def test_when_called_then_returns_self(self):
#         # Arrange
#         module = _DoubleModule()

#         # Act
#         actual = module.eval()

#         # Assert
#         self.assertIs(actual, module)


# class TestTraining(unittest.TestCase):
#     def test_when_constructed_then_defaults_to_true(self):
#         # Arrange
#         module = _DoubleModule()

#         # Act
#         actual = module.training

#         # Assert
#         self.assertTrue(actual)

#     def test_when_one_module_is_set_to_eval_then_another_instance_still_defaults_to_true(
#             self):
#         # Arrange
#         first = _DoubleModule()
#         second = _DoubleModule()
#         first.eval()

#         # Act
#         actual = second.training

#         # Assert
#         self.assertTrue(actual)
