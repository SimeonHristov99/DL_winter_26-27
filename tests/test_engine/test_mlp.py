# import random
# import unittest
#
# from dl_lib.engine.mlp import MLP, Layer, Neuron
# from dl_lib.engine.value import Value
#
#
# class TestNeuronInit(unittest.TestCase):
#
#     def test_when_constructed_then_has_one_weight_per_input(self):
#         # Arrange
#         expected = 3
#
#         # Act
#         actual = len(Neuron(3).w)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_with_seeded_rng_then_weights_match_uniform_draws(
#             self):
#         # Arrange
#         rng = random.Random(42)
#         expected = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
#
#         # Act
#         actual = [w.data for w in Neuron(2, rng=random.Random(42)).w]
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_with_seeded_rng_then_bias_matches_next_uniform_draw(
#             self):
#         # Arrange
#         rng = random.Random(42)
#         rng.uniform(-1, 1)
#         rng.uniform(-1, 1)
#         expected = rng.uniform(-1, 1)
#
#         # Act
#         actual = Neuron(2, rng=random.Random(42)).b.data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestNeuronCall(unittest.TestCase):
#
#     def test_when_called_then_output_matches_hand_calculation(self):
#         # Arrange
#         neuron = Neuron(2, rng=random.Random(42))
#         x = [Value(1.0), Value(2.0)]
#         expected = -0.968717830476061
#
#         # Act
#         actual = neuron(x).data
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_output_is_a_value(self):
#         # Arrange
#         neuron = Neuron(2)
#         x = [Value(1.0), Value(2.0)]
#
#         # Act
#         actual = neuron(x)
#
#         # Assert
#         self.assertIsInstance(actual, Value)
#
#
# class TestNeuronParameters(unittest.TestCase):
#
#     def test_when_parameters_called_then_returns_weights_then_bias(self):
#         # Arrange
#         neuron = Neuron(2)
#         expected = [*neuron.w, neuron.b]
#
#         # Act
#         actual = neuron.parameters()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_parameters_called_then_count_matches_inputs_plus_bias(self):
#         # Arrange
#         neuron = Neuron(5)
#         expected = 6
#
#         # Act
#         actual = len(neuron.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestLayerInit(unittest.TestCase):
#
#     def test_when_constructed_then_has_one_neuron_per_output(self):
#         # Arrange
#         expected = 4
#
#         # Act
#         actual = len(Layer(3, 4).neurons)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_with_shared_rng_then_neurons_consume_sequential_draws(
#             self):
#         # Arrange
#         rng = random.Random(42)
#         expected = []
#         for _ in range(2):
#             expected.append([rng.uniform(-1, 1) for _ in range(3)])
#             expected.append(rng.uniform(-1, 1))
#
#         # Act
#         layer = Layer(3, 2, rng=random.Random(42))
#         actual = []
#         for neuron in layer.neurons:
#             actual.append([v.data for v in neuron.w])
#             actual.append(neuron.b.data)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestLayerCall(unittest.TestCase):
#
#     def test_when_called_then_returns_one_output_per_neuron(self):
#         # Arrange
#         layer = Layer(3, 4)
#         x = [Value(1.0), Value(2.0), Value(3.0)]
#         expected = 4
#
#         # Act
#         actual = len(layer(x))
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_on_single_neuron_layer_then_still_returns_a_list(
#             self):
#         # Arrange
#         layer = Layer(3, 1)
#         x = [Value(1.0), Value(2.0), Value(3.0)]
#
#         # Act
#         actual = layer(x)
#
#         # Assert
#         self.assertIsInstance(actual, list)
#         self.assertEqual(len(actual), 1)
#
#
# class TestLayerParameters(unittest.TestCase):
#
#     def test_when_parameters_called_then_returns_parameters_from_every_neuron(
#             self):
#         # Arrange
#         layer = Layer(3, 2)
#         expected = [
#             *layer.neurons[0].parameters(), *layer.neurons[1].parameters()
#         ]
#
#         # Act
#         actual = layer.parameters()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_parameters_called_then_count_matches_neurons_times_their_size(
#             self):
#         # Arrange
#         layer = Layer(3, 4)
#         expected = 4 * (3 + 1)
#
#         # Act
#         actual = len(layer.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestMLPInit(unittest.TestCase):
#
#     def test_when_constructed_then_has_one_layer_per_output_size(self):
#         # Arrange
#         expected = 2
#
#         # Act
#         actual = len(MLP(3, [4, 1]).layers)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_then_each_layers_input_size_matches_previous_output(
#             self):
#         # Arrange
#         mlp = MLP(3, [4, 2, 1])
#         expected = [3, 4, 2]
#
#         # Act
#         actual = [len(layer.neurons[0].w) for layer in mlp.layers]
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestMLPCall(unittest.TestCase):
#
#     def test_when_called_with_multi_neuron_output_layer_then_returns_a_list(
#             self):
#         # Arrange
#         mlp = MLP(3, [4, 2])
#         x = [Value(1.0), Value(2.0), Value(3.0)]
#         expected = 2
#
#         # Act
#         actual = mlp(x)
#
#         # Assert
#         self.assertIsInstance(actual, list)
#         self.assertEqual(len(actual), expected)
#
#     def test_when_called_with_single_neuron_output_layer_then_returns_a_bare_value(
#             self):
#         # Arrange
#         mlp = MLP(3, [4, 1])
#         x = [Value(1.0), Value(2.0), Value(3.0)]
#
#         # Act
#         actual = mlp(x)
#
#         # Assert
#         self.assertIsInstance(actual, Value)
#
#
# class TestMLPParameters(unittest.TestCase):
#
#     def test_when_parameters_called_then_returns_parameters_from_every_layer(
#             self):
#         # Arrange
#         mlp = MLP(3, [4, 1])
#         expected = [*mlp.layers[0].parameters(), *mlp.layers[1].parameters()]
#
#         # Act
#         actual = mlp.parameters()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_parameters_called_then_count_matches_sum_of_layer_counts(
#             self):
#         # Arrange
#         mlp = MLP(3, [4, 4, 1])
#         expected = sum(len(layer.parameters()) for layer in mlp.layers)
#
#         # Act
#         actual = len(mlp.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
