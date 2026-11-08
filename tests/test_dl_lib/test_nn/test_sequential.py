# import unittest

# import torch

# from dl_lib.nn.activations import ReLU, Sigmoid, Tanh
# from dl_lib.nn.module import Module
# from dl_lib.nn.sequential import Sequential


# class _ParamModule(Module):
#     def __init__(self, param: torch.Tensor) -> None:
#         self.param = param

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x

#     def parameters(self) -> "list[torch.Tensor]":
#         return [self.param]


# class TestInit(unittest.TestCase):
#     def test_when_constructed_then_stores_the_given_modules_in_order(self):
#         # Arrange
#         relu = ReLU()
#         tanh = Tanh()
#         expected = [relu, tanh]

#         # Act
#         actual = Sequential(relu, tanh).modules

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_constructed_with_no_modules_then_modules_list_is_empty(self):
#         # Arrange
#         expected = []

#         # Act
#         actual = Sequential().modules

#         # Assert
#         self.assertEqual(actual, expected)


# class TestForward(unittest.TestCase):
#     def test_when_called_then_chains_modules_in_order(self):
#         # Arrange
#         sequential = Sequential(ReLU(), Tanh())
#         x = torch.tensor([-2.0, 2.0])
#         expected = torch.tanh(torch.tensor([0.0, 2.0]))

#         # Act
#         actual = sequential(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))

#     def test_when_modules_reordered_then_output_reflects_the_new_order(self):
#         # Arrange
#         sequential = Sequential(Tanh(), ReLU())
#         x = torch.tensor([-2.0, 2.0])
#         expected = torch.max(torch.tanh(x), torch.zeros_like(x))

#         # Act
#         actual = sequential(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual, expected))


# class TestAppend(unittest.TestCase):
#     def test_when_called_then_adds_module_to_the_end(self):
#         # Arrange
#         sequential = Sequential(ReLU())
#         expected = 2

#         # Act
#         sequential.append(Tanh())

#         # Assert
#         self.assertEqual(len(sequential.modules), expected)
#         self.assertIsInstance(sequential.modules[-1], Tanh)

#     def test_when_called_twice_then_both_modules_are_kept_in_order(self):
#         # Arrange
#         sequential = Sequential(ReLU())
#         expected = [ReLU, Tanh, Sigmoid]

#         # Act
#         sequential.append(Tanh())
#         sequential.append(Sigmoid())

#         # Assert
#         self.assertEqual([type(module) for module in sequential.modules], expected)


# class TestExtend(unittest.TestCase):
#     def test_when_called_then_adds_every_module_in_order(self):
#         # Arrange
#         sequential = Sequential(ReLU())
#         expected = [ReLU, Tanh, Sigmoid]

#         # Act
#         sequential.extend([Tanh(), Sigmoid()])

#         # Assert
#         self.assertEqual([type(module) for module in sequential.modules], expected)

#     def test_when_extended_with_empty_list_then_modules_are_unchanged(self):
#         # Arrange
#         sequential = Sequential(ReLU())
#         expected = [ReLU]

#         # Act
#         sequential.extend([])

#         # Assert
#         self.assertEqual([type(module) for module in sequential.modules], expected)


# class TestInsert(unittest.TestCase):
#     def test_when_called_then_places_module_at_the_given_index(self):
#         # Arrange
#         sequential = Sequential(ReLU(), Sigmoid())
#         expected = [ReLU, Tanh, Sigmoid]

#         # Act
#         sequential.insert(1, Tanh())

#         # Assert
#         self.assertEqual([type(module) for module in sequential.modules], expected)

#     def test_when_called_with_index_zero_then_places_module_first(self):
#         # Arrange
#         sequential = Sequential(ReLU(), Sigmoid())
#         expected = [Tanh, ReLU, Sigmoid]

#         # Act
#         sequential.insert(0, Tanh())

#         # Assert
#         self.assertEqual([type(module) for module in sequential.modules], expected)


# class TestParameters(unittest.TestCase):
#     def test_when_called_then_collects_parameters_from_every_module_in_order(self):
#         # Arrange
#         first = torch.tensor([1.0])
#         second = torch.tensor([2.0])
#         sequential = Sequential(_ParamModule(first), ReLU(), _ParamModule(second))
#         expected = [first, second]

#         # Act
#         actual = sequential.parameters()

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_no_module_owns_parameters_then_returns_empty_list(self):
#         # Arrange
#         sequential = Sequential(ReLU(), Tanh())
#         expected = []

#         # Act
#         actual = sequential.parameters()

#         # Assert
#         self.assertEqual(actual, expected)


# class TestTrain(unittest.TestCase):
#     def test_when_called_then_sets_every_module_to_training_mode(self):
#         # Arrange
#         relu, tanh = ReLU(), Tanh()
#         sequential = Sequential(relu, tanh)
#         sequential.eval()

#         # Act
#         sequential.train()

#         # Assert
#         self.assertTrue(sequential.training)
#         self.assertTrue(relu.training)
#         self.assertTrue(tanh.training)

#     def test_when_nested_then_propagates_into_the_inner_sequential(self):
#         # Arrange
#         inner = Sequential(ReLU())
#         outer = Sequential(inner, Tanh())
#         outer.eval()

#         # Act
#         outer.train()

#         # Assert
#         self.assertTrue(inner.training)
#         self.assertTrue(inner.modules[0].training)


# class TestEval(unittest.TestCase):
#     def test_when_called_then_sets_every_module_to_evaluation_mode(self):
#         # Arrange
#         relu, tanh = ReLU(), Tanh()
#         sequential = Sequential(relu, tanh)

#         # Act
#         sequential.eval()

#         # Assert
#         self.assertFalse(sequential.training)
#         self.assertFalse(relu.training)
#         self.assertFalse(tanh.training)

#     def test_when_nested_then_propagates_into_the_inner_sequential(self):
#         # Arrange
#         inner = Sequential(ReLU())
#         outer = Sequential(inner, Tanh())

#         # Act
#         outer.eval()

#         # Assert
#         self.assertFalse(inner.training)
#         self.assertFalse(inner.modules[0].training)
