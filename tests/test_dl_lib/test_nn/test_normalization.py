# import unittest

# import torch
# from dl_lib.nn.normalization import BatchNorm1d, LayerNorm


# def _make_input():
#     return torch.tensor(
#         [
#             [1.0, 10.0, -2.0],
#             [3.0, 14.0, 4.0],
#             [5.0, 18.0, 1.0],
#             [2.0, 12.0, 8.0],
#         ]
#     )


# def _make_nearly_constant_input():
#     return torch.tensor(
#         [
#             [1.00, 1.00, 1.00],
#             [1.01, 1.02, 1.03],
#             [1.04, 1.01, 1.02],
#             [1.02, 1.03, 1.01],
#         ]
#     )


# def _make_batchnorm_pair():
#     ours = BatchNorm1d(num_features=3)
#     reference = torch.nn.BatchNorm1d(num_features=3)

#     with torch.no_grad():
#         ours.gamma.copy_(torch.tensor([2.0, -0.5, 0.25]))
#         ours.beta.copy_(torch.tensor([1.0, 3.0, -2.0]))
#         ours.running_mean.copy_(torch.tensor([2.0, 8.0, -1.0]))
#         ours.running_var.copy_(torch.tensor([4.0, 9.0, 2.0]))
#         reference.weight.copy_(ours.gamma)
#         reference.bias.copy_(ours.beta)
#         reference.running_mean.copy_(ours.running_mean)
#         reference.running_var.copy_(ours.running_var)

#     return ours, reference


# def _make_layernorm_pair():
#     ours = LayerNorm(normalized_shape=3)
#     reference = torch.nn.LayerNorm(normalized_shape=3)

#     with torch.no_grad():
#         ours.gamma.copy_(torch.tensor([2.0, -0.5, 0.25]))
#         ours.beta.copy_(torch.tensor([1.0, 3.0, -2.0]))
#         reference.weight.copy_(ours.gamma)
#         reference.bias.copy_(ours.beta)

#     return ours, reference


# def _make_backward_inputs():
#     x = _make_input().requires_grad_()
#     reference_x = x.detach().clone().requires_grad_()
#     upstream = torch.tensor(
#         [
#             [0.2, -1.0, 3.0],
#             [1.5, 0.3, -0.7],
#             [-2.0, 0.8, 1.2],
#             [0.4, -0.6, 2.1],
#         ]
#     )
#     return x, reference_x, upstream


# def _assert_gradients_match(test_case, ours, reference, x, reference_x):
#     test_case.assertIsNotNone(x.grad)
#     test_case.assertIsNotNone(ours.gamma.grad)
#     test_case.assertIsNotNone(ours.beta.grad)

#     torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-5, rtol=1e-4)
#     torch.testing.assert_close(
#         ours.gamma.grad, reference.weight.grad, atol=1e-5, rtol=1e-4
#     )
#     torch.testing.assert_close(
#         ours.beta.grad, reference.bias.grad, atol=1e-5, rtol=1e-4
#     )


# def _make_batches_with_fixed_example():
#     fixed = torch.tensor([[1.0, 3.0, 8.0]])
#     batch_a = torch.cat([fixed, torch.tensor([[2.0, 4.0, 6.0]])])
#     batch_b = torch.cat(
#         [
#             fixed,
#             torch.tensor(
#                 [
#                     [100.0, -50.0, 20.0],
#                     [-30.0, 80.0, 200.0],
#                 ]
#             ),
#         ]
#     )
#     return fixed, batch_a, batch_b


# class TestBatchNorm1dInit(unittest.TestCase):
#     def test_when_constructed_then_gamma_is_ones_and_beta_is_zeros(self):
#         # Arrange
#         num_features = 5

#         # Act
#         batch_norm = BatchNorm1d(num_features)

#         # Assert
#         self.assertTrue(torch.equal(batch_norm.gamma, torch.ones(num_features)))
#         self.assertTrue(torch.equal(batch_norm.beta, torch.zeros(num_features)))

#     def test_when_constructed_then_running_mean_is_zeros_and_running_var_is_ones(
#         self,
#     ):
#         # Arrange
#         num_features = 5

#         # Act
#         batch_norm = BatchNorm1d(num_features)

#         # Assert
#         self.assertTrue(torch.equal(batch_norm.running_mean, torch.zeros(num_features)))
#         self.assertTrue(torch.equal(batch_norm.running_var, torch.ones(num_features)))

#     def test_when_eps_and_momentum_not_given_then_default_to_1e_minus_5_and_0_1(
#         self,
#     ):
#         # Arrange
#         expected_eps = 1e-5
#         expected_momentum = 0.1

#         # Act
#         batch_norm = BatchNorm1d(num_features=3)

#         # Assert
#         self.assertEqual(batch_norm.eps, expected_eps)
#         self.assertEqual(batch_norm.momentum, expected_momentum)


# class TestBatchNorm1dForward(unittest.TestCase):
#     def test_when_training_then_output_is_zero_mean_and_unit_variance_per_feature(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         batch_norm = BatchNorm1d(num_features=4)
#         x = torch.randn(32, 4) * 5.0 + 3.0

#         # Act
#         actual = batch_norm(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual.mean(dim=0), torch.zeros(4), atol=1e-5))
#         self.assertTrue(
#             torch.allclose(
#                 actual.var(dim=0, unbiased=False),
#                 torch.ones(4),
#                 atol=1e-4,
#             )
#         )

#     def test_when_evaluating_then_uses_stored_running_statistics(self):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=3)
#         with torch.no_grad():
#             batch_norm.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
#             batch_norm.running_var.copy_(torch.tensor([4.0, 9.0, 16.0]))
#         batch_norm.eval()
#         x = torch.tensor([[3.0, 8.0, -1.0]])
#         expected = (x - batch_norm.running_mean) / torch.sqrt(
#             batch_norm.running_var + batch_norm.eps
#         )

#         # Act
#         actual = batch_norm(x)

#         # Assert
#         torch.testing.assert_close(actual, expected)

#     def test_when_training_then_updates_running_statistics_using_unbiased_variance(
#         self,
#     ):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=2, momentum=0.1)
#         x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
#         expected_mean = 0.9 * torch.zeros(2) + 0.1 * x.mean(dim=0)
#         expected_var = 0.9 * torch.ones(2) + 0.1 * x.var(dim=0, unbiased=True)

#         # Act
#         batch_norm(x)

#         # Assert
#         torch.testing.assert_close(batch_norm.running_mean, expected_mean)
#         torch.testing.assert_close(batch_norm.running_var, expected_var)

#     def test_when_given_same_weights_then_matches_torch_nn_batchnorm1d(self):
#         # Arrange
#         torch.manual_seed(0)
#         x = torch.randn(16, 4)
#         ours = BatchNorm1d(num_features=4)
#         reference = torch.nn.BatchNorm1d(num_features=4)

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_evaluating_then_matches_torch_nn_batchnorm1d(self):
#         # Arrange
#         torch.manual_seed(0)
#         x_train = torch.randn(16, 4)
#         x_eval = torch.randn(8, 4)
#         ours = BatchNorm1d(num_features=4)
#         reference = torch.nn.BatchNorm1d(num_features=4)
#         ours(x_train)
#         reference(x_train)
#         ours.eval()
#         reference.eval()

#         # Act
#         actual = ours(x_eval)
#         expected = reference(x_eval)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_training_with_nondefault_gamma_and_beta_then_matches_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_batchnorm_pair()
#         x = _make_input()
#         ours.train()
#         reference.train()

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_evaluating_with_nondefault_gamma_and_beta_then_matches_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_batchnorm_pair()
#         x = _make_input()
#         ours.eval()
#         reference.eval()

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_training_on_multiple_batches_then_running_statistics_accumulate(
#         self,
#     ):
#         # Arrange
#         momentum = 0.25
#         batch_norm = BatchNorm1d(num_features=2, momentum=momentum)
#         batches = [
#             torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]]),
#             torch.tensor([[-4.0, 10.0], [0.0, 14.0], [2.0, 18.0]]),
#             torch.tensor([[8.0, -3.0], [12.0, 1.0], [16.0, 5.0]]),
#         ]
#         expected_mean = torch.zeros(2)
#         expected_var = torch.ones(2)

#         for batch in batches:
#             expected_mean = (1 - momentum) * expected_mean + momentum * batch.mean(
#                 dim=0
#             )
#             expected_var = (1 - momentum) * expected_var + momentum * batch.var(
#                 dim=0, unbiased=True
#             )

#         # Act
#         for batch in batches:
#             batch_norm(batch)

#         # Assert
#         torch.testing.assert_close(batch_norm.running_mean, expected_mean)
#         torch.testing.assert_close(batch_norm.running_var, expected_var)

#     def test_when_evaluating_multiple_batches_then_statistics_and_parameters_are_unchanged(
#         self,
#     ):
#         # Arrange
#         batch_norm, _ = _make_batchnorm_pair()
#         batch_norm(_make_input())
#         expected_mean = batch_norm.running_mean.clone()
#         expected_var = batch_norm.running_var.clone()
#         expected_gamma = batch_norm.gamma.detach().clone()
#         expected_beta = batch_norm.beta.detach().clone()
#         batch_norm.eval()

#         # Act
#         batch_norm(torch.tensor([[100.0, -20.0, 5.0], [200.0, -40.0, 8.0]]))
#         batch_norm(torch.tensor([[-500.0, 300.0, -10.0]]))

#         # Assert
#         self.assertTrue(torch.equal(batch_norm.running_mean, expected_mean))
#         self.assertTrue(torch.equal(batch_norm.running_var, expected_var))
#         self.assertTrue(torch.equal(batch_norm.gamma, expected_gamma))
#         self.assertTrue(torch.equal(batch_norm.beta, expected_beta))

#     def test_when_input_is_constant_then_output_is_finite_and_zero(self):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=3)
#         x = torch.full((4, 3), 2.0)
#         expected = torch.zeros_like(x)

#         # Act
#         actual = batch_norm(x)

#         # Assert
#         self.assertTrue(torch.isfinite(actual).all().item())
#         torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

#     def test_when_eps_is_custom_then_uses_it_inside_square_root(self):
#         # Arrange
#         eps = 0.25
#         batch_norm = BatchNorm1d(num_features=3, eps=eps)
#         x = _make_nearly_constant_input()
#         mean = x.mean(dim=0, keepdim=True)
#         variance = x.var(dim=0, unbiased=False, keepdim=True)
#         expected = (x - mean) / torch.sqrt(variance + eps)

#         # Act
#         actual = batch_norm(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

#     def test_when_backpropagating_during_training_then_gradients_match_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_batchnorm_pair()
#         x, reference_x, upstream = _make_backward_inputs()
#         ours.train()
#         reference.train()

#         # Act
#         ours(x).backward(upstream)
#         reference(reference_x).backward(upstream)

#         # Assert
#         _assert_gradients_match(self, ours, reference, x, reference_x)

#     def test_when_backpropagating_during_evaluation_then_gradients_match_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_batchnorm_pair()
#         x, reference_x, upstream = _make_backward_inputs()
#         ours.eval()
#         reference.eval()

#         # Act
#         ours(x).backward(upstream)
#         reference(reference_x).backward(upstream)

#         # Assert
#         _assert_gradients_match(self, ours, reference, x, reference_x)

#     def test_when_training_with_differentiable_inputs_then_running_statistics_do_not_track_gradients(
#         self,
#     ):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=2)
#         x = torch.tensor(
#             [[1.0, 2.0], [3.0, 8.0], [5.0, 4.0]],
#             requires_grad=True,
#         )

#         # Act
#         batch_norm(x)
#         batch_norm(x * 2.0)

#         # Assert
#         self.assertFalse(batch_norm.running_mean.requires_grad)
#         self.assertFalse(batch_norm.running_var.requires_grad)
#         self.assertIsNone(batch_norm.running_mean.grad_fn)
#         self.assertIsNone(batch_norm.running_var.grad_fn)

#     def test_when_evaluating_and_batchmates_change_then_fixed_example_output_is_unchanged(
#         self,
#     ):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=3)
#         batch_norm(_make_input())
#         batch_norm.eval()
#         fixed, batch_a, batch_b = _make_batches_with_fixed_example()

#         # Act
#         alone = batch_norm(fixed)[0]
#         with_batch_a = batch_norm(batch_a)[0]
#         with_batch_b = batch_norm(batch_b)[0]

#         # Assert
#         torch.testing.assert_close(alone, with_batch_a)
#         torch.testing.assert_close(alone, with_batch_b)


# class TestBatchNorm1dParameters(unittest.TestCase):
#     def test_when_called_then_returns_gamma_and_beta(self):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=3)
#         expected = [batch_norm.gamma, batch_norm.beta]

#         # Act
#         actual = batch_norm.parameters()

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_gamma_and_beta_are_updated_then_returns_their_current_values(
#         self,
#     ):
#         # Arrange
#         batch_norm = BatchNorm1d(num_features=2)
#         with torch.no_grad():
#             batch_norm.gamma += 1.0
#             batch_norm.beta += 1.0
#         expected = [batch_norm.gamma, batch_norm.beta]

#         # Act
#         actual = batch_norm.parameters()

#         # Assert
#         self.assertEqual(actual, expected)


# class TestLayerNormInit(unittest.TestCase):
#     def test_when_constructed_then_gamma_is_ones_and_beta_is_zeros(self):
#         # Arrange
#         normalized_shape = 6

#         # Act
#         layer_norm = LayerNorm(normalized_shape)

#         # Assert
#         self.assertTrue(torch.equal(layer_norm.gamma, torch.ones(normalized_shape)))
#         self.assertTrue(torch.equal(layer_norm.beta, torch.zeros(normalized_shape)))

#     def test_when_eps_not_given_then_defaults_to_1e_minus_5(self):
#         # Arrange
#         normalized_shape = 4
#         expected = 1e-5

#         # Act
#         actual = LayerNorm(normalized_shape).eps

#         # Assert
#         self.assertEqual(actual, expected)


# class TestLayerNormForward(unittest.TestCase):
#     def test_when_called_then_output_is_zero_mean_and_unit_variance_per_sample(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(0)
#         layer_norm = LayerNorm(normalized_shape=8)
#         x = torch.randn(4, 8) * 3.0 + 1.0

#         # Act
#         actual = layer_norm(x)

#         # Assert
#         self.assertTrue(torch.allclose(actual.mean(dim=-1), torch.zeros(4), atol=1e-5))
#         self.assertTrue(
#             torch.allclose(
#                 actual.var(dim=-1, unbiased=False),
#                 torch.ones(4),
#                 atol=1e-4,
#             )
#         )

#     def test_when_training_or_evaluating_then_behavior_is_identical(self):
#         # Arrange
#         torch.manual_seed(0)
#         layer_norm = LayerNorm(normalized_shape=6)
#         x = torch.randn(2, 6)
#         layer_norm.train()
#         expected = layer_norm(x)
#         layer_norm.eval()

#         # Act
#         actual = layer_norm(x)

#         # Assert
#         self.assertTrue(torch.equal(actual, expected))

#     def test_when_given_same_weights_then_matches_torch_nn_layernorm(self):
#         # Arrange
#         torch.manual_seed(0)
#         x = torch.randn(4, 5)
#         ours = LayerNorm(normalized_shape=5)
#         reference = torch.nn.LayerNorm(normalized_shape=5)

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_input_has_an_extra_leading_dimension_then_still_matches_torch_nn_layernorm(
#         self,
#     ):
#         # Arrange
#         torch.manual_seed(1)
#         x = torch.randn(2, 3, 6)
#         ours = LayerNorm(normalized_shape=6)
#         reference = torch.nn.LayerNorm(normalized_shape=6)

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_training_with_nondefault_gamma_and_beta_then_matches_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_layernorm_pair()
#         x = _make_input()
#         ours.train()
#         reference.train()

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_evaluating_with_nondefault_gamma_and_beta_then_matches_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_layernorm_pair()
#         x = _make_input()
#         ours.eval()
#         reference.eval()

#         # Act
#         actual = ours(x)
#         expected = reference(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

#     def test_when_input_is_constant_then_output_is_finite_and_zero(self):
#         # Arrange
#         layer_norm = LayerNorm(normalized_shape=3)
#         x = torch.full((4, 3), 2.0)
#         expected = torch.zeros_like(x)

#         # Act
#         actual = layer_norm(x)

#         # Assert
#         self.assertTrue(torch.isfinite(actual).all().item())
#         torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

#     def test_when_eps_is_custom_then_uses_it_inside_square_root(self):
#         # Arrange
#         eps = 0.25
#         layer_norm = LayerNorm(normalized_shape=3, eps=eps)
#         x = _make_nearly_constant_input()
#         mean = x.mean(dim=-1, keepdim=True)
#         variance = x.var(dim=-1, unbiased=False, keepdim=True)
#         expected = (x - mean) / torch.sqrt(variance + eps)

#         # Act
#         actual = layer_norm(x)

#         # Assert
#         torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

#     def test_when_backpropagating_during_training_then_gradients_match_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_layernorm_pair()
#         x, reference_x, upstream = _make_backward_inputs()
#         ours.train()
#         reference.train()

#         # Act
#         ours(x).backward(upstream)
#         reference(reference_x).backward(upstream)

#         # Assert
#         _assert_gradients_match(self, ours, reference, x, reference_x)

#     def test_when_backpropagating_during_evaluation_then_gradients_match_torch(
#         self,
#     ):
#         # Arrange
#         ours, reference = _make_layernorm_pair()
#         x, reference_x, upstream = _make_backward_inputs()
#         ours.eval()
#         reference.eval()

#         # Act
#         ours(x).backward(upstream)
#         reference(reference_x).backward(upstream)

#         # Assert
#         _assert_gradients_match(self, ours, reference, x, reference_x)

#     def test_when_training_and_batchmates_change_then_fixed_example_output_is_unchanged(
#         self,
#     ):
#         # Arrange
#         layer_norm = LayerNorm(normalized_shape=3)
#         fixed, batch_a, batch_b = _make_batches_with_fixed_example()
#         layer_norm.train()

#         # Act
#         alone = layer_norm(fixed)[0]
#         with_batch_a = layer_norm(batch_a)[0]
#         with_batch_b = layer_norm(batch_b)[0]

#         # Assert
#         torch.testing.assert_close(alone, with_batch_a)
#         torch.testing.assert_close(alone, with_batch_b)

#     def test_when_evaluating_and_batchmates_change_then_fixed_example_output_is_unchanged(
#         self,
#     ):
#         # Arrange
#         layer_norm = LayerNorm(normalized_shape=3)
#         fixed, batch_a, batch_b = _make_batches_with_fixed_example()
#         layer_norm.eval()

#         # Act
#         alone = layer_norm(fixed)[0]
#         with_batch_a = layer_norm(batch_a)[0]
#         with_batch_b = layer_norm(batch_b)[0]

#         # Assert
#         torch.testing.assert_close(alone, with_batch_a)
#         torch.testing.assert_close(alone, with_batch_b)


# class TestLayerNormParameters(unittest.TestCase):
#     def test_when_called_then_returns_gamma_and_beta(self):
#         # Arrange
#         layer_norm = LayerNorm(normalized_shape=4)
#         expected = [layer_norm.gamma, layer_norm.beta]

#         # Act
#         actual = layer_norm.parameters()

#         # Assert
#         self.assertEqual(actual, expected)

#     def test_when_gamma_and_beta_are_updated_then_returns_their_current_values(
#         self,
#     ):
#         # Arrange
#         layer_norm = LayerNorm(normalized_shape=4)
#         with torch.no_grad():
#             layer_norm.gamma += 1.0
#             layer_norm.beta += 1.0
#         expected = [layer_norm.gamma, layer_norm.beta]

#         # Act
#         actual = layer_norm.parameters()

#         # Assert
#         self.assertEqual(actual, expected)
