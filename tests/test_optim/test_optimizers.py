import unittest

import torch

from dl_lib.optim.optimizers import SGD, AdaGrad, Adam, AdamW, RMSprop


class TestSGDInit(unittest.TestCase):
    def test_when_momentum_not_given_then_defaults_to_zero(self):
        # Arrange
        p = torch.tensor([1.0])
        expected = 0.0

        # Act
        actual = SGD([p], lr=0.1).momentum

        # Assert
        self.assertEqual(actual, expected)

    def test_when_constructed_then_velocities_are_zero_for_every_parameter(self):
        # Arrange
        p1 = torch.tensor([1.0, 2.0])
        p2 = torch.tensor([3.0])
        expected = [torch.zeros_like(p1), torch.zeros_like(p2)]

        # Act
        actual = SGD([p1, p2], lr=0.1).velocities

        # Assert
        self.assertEqual(len(actual), len(expected))
        for actual_velocity, expected_velocity in zip(actual, expected):
            self.assertTrue(torch.equal(actual_velocity, expected_velocity))


class TestSGDStep(unittest.TestCase):
    def test_when_called_then_updates_parameter_by_minus_lr_times_grad(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        p.grad = torch.tensor([0.1, 0.2])
        lr = 0.1
        sgd = SGD([p], lr=lr)
        expected = torch.tensor([0.99, 1.98])

        # Act
        sgd.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected))

    def test_when_called_with_a_different_learning_rate_then_scales_the_update_accordingly(
        self,
    ):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        p.grad = torch.tensor([0.1, 0.2])
        lr = 0.5
        sgd = SGD([p], lr=lr)
        expected = torch.tensor([0.95, 1.9])

        # Act
        sgd.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected))


class TestSGDMomentumStep(unittest.TestCase):
    def test_when_called_twice_then_velocity_accumulates_across_steps(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        lr = 0.1
        momentum = 0.9
        sgd = SGD([p], lr=lr, momentum=momentum)
        expected_velocity_after_first_step = torch.tensor([-0.01, -0.02])
        expected_velocity_after_second_step = torch.tensor([-0.019, -0.038])

        # Act
        p.grad = torch.tensor([0.1, 0.2])
        sgd.step()
        actual_velocity_after_first_step = sgd.velocities[0].clone()

        p.grad = torch.tensor([0.1, 0.2])
        sgd.step()
        actual_velocity_after_second_step = sgd.velocities[0].clone()

        # Assert
        self.assertTrue(
            torch.allclose(
                actual_velocity_after_first_step, expected_velocity_after_first_step
            )
        )
        self.assertTrue(
            torch.allclose(
                actual_velocity_after_second_step, expected_velocity_after_second_step
            )
        )

    def test_when_called_twice_then_parameter_matches_hand_computed_trajectory(self):
        # Arrange
        p = torch.tensor([1.0, 2.0], requires_grad=True)
        lr = 0.1
        momentum = 0.9
        sgd = SGD([p], lr=lr, momentum=momentum)
        expected = torch.tensor([0.971, 1.942])

        # Act
        p.grad = torch.tensor([0.1, 0.2])
        sgd.step()
        p.grad = torch.tensor([0.1, 0.2])
        sgd.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected))


class TestAdaGradInit(unittest.TestCase):
    def test_when_eps_not_given_then_defaults_to_1e_minus_10(self):
        # Arrange
        p = torch.tensor([1.0])
        expected = 1e-10

        # Act
        actual = AdaGrad([p], lr=0.1).eps

        # Assert
        self.assertEqual(actual, expected)

    def test_when_constructed_then_accumulators_are_zero_for_every_parameter(self):
        # Arrange
        p1 = torch.tensor([1.0, 2.0])
        p2 = torch.tensor([3.0])
        expected = [torch.zeros_like(p1), torch.zeros_like(p2)]

        # Act
        actual = AdaGrad([p1, p2], lr=0.1).accumulators

        # Assert
        self.assertEqual(len(actual), len(expected))
        for actual_accumulator, expected_accumulator in zip(actual, expected):
            self.assertTrue(torch.equal(actual_accumulator, expected_accumulator))


class TestAdaGradStep(unittest.TestCase):
    def test_when_called_then_accumulator_holds_squared_gradient(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        adagrad = AdaGrad([p], lr=0.1)
        expected = torch.tensor([0.01])

        # Act
        adagrad.step()

        # Assert
        self.assertTrue(torch.allclose(adagrad.accumulators[0], expected))

    def test_when_called_then_updates_parameter_with_per_parameter_learning_rate(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        adagrad = AdaGrad([p], lr=0.1)
        expected = torch.tensor([0.9])

        # Act
        adagrad.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected, atol=1e-4))


class TestRMSpropInit(unittest.TestCase):
    def test_when_alpha_and_eps_not_given_then_default_to_0_99_and_1e_minus_8(self):
        # Arrange
        p = torch.tensor([1.0])
        expected_alpha = 0.99
        expected_eps = 1e-8

        # Act
        rmsprop = RMSprop([p], lr=0.1)

        # Assert
        self.assertEqual(rmsprop.alpha, expected_alpha)
        self.assertEqual(rmsprop.eps, expected_eps)

    def test_when_constructed_then_accumulators_are_zero_for_every_parameter(self):
        # Arrange
        p1 = torch.tensor([1.0, 2.0])
        p2 = torch.tensor([3.0])
        expected = [torch.zeros_like(p1), torch.zeros_like(p2)]

        # Act
        actual = RMSprop([p1, p2], lr=0.1).accumulators

        # Assert
        self.assertEqual(len(actual), len(expected))
        for actual_accumulator, expected_accumulator in zip(actual, expected):
            self.assertTrue(torch.equal(actual_accumulator, expected_accumulator))


class TestRMSpropStep(unittest.TestCase):
    def test_when_called_then_accumulator_is_an_exponential_moving_average(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        rmsprop = RMSprop([p], lr=0.1)
        expected = torch.tensor([0.0001])

        # Act
        rmsprop.step()

        # Assert
        self.assertTrue(torch.allclose(rmsprop.accumulators[0], expected))

    def test_when_called_then_updates_parameter_with_per_parameter_learning_rate(self):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        rmsprop = RMSprop([p], lr=0.1)
        expected = torch.tensor([0.0])

        # Act
        rmsprop.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected, atol=1e-4))


class TestAdamInit(unittest.TestCase):
    def test_when_optional_args_not_given_then_default_values_are_used(self):
        # Arrange
        p = torch.tensor([1.0])
        expected_beta1 = 0.9
        expected_beta2 = 0.999
        expected_eps = 1e-8
        expected_weight_decay = 0.0

        # Act
        adam = Adam([p], lr=0.001)

        # Assert
        self.assertEqual(adam.beta1, expected_beta1)
        self.assertEqual(adam.beta2, expected_beta2)
        self.assertEqual(adam.eps, expected_eps)
        self.assertEqual(adam.weight_decay, expected_weight_decay)

    def test_when_constructed_then_moments_are_zero_and_step_count_is_zero(self):
        # Arrange
        p = torch.tensor([1.0, 2.0])
        expected_moment = torch.zeros_like(p)
        expected_t = 0

        # Act
        adam = Adam([p], lr=0.001)

        # Assert
        self.assertTrue(torch.equal(adam.m[0], expected_moment))
        self.assertTrue(torch.equal(adam.v[0], expected_moment))
        self.assertEqual(adam.t, expected_t)


class TestAdamStep(unittest.TestCase):
    def test_when_called_once_then_matches_the_bias_corrected_update_at_t_equals_one(
        self,
    ):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        adam = Adam([p], lr=0.001)
        expected = torch.tensor([0.999])

        # Act
        adam.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected, atol=1e-4))

    def test_when_called_twice_then_matches_the_bias_corrected_update_at_t_equals_two(
        self,
    ):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        adam = Adam([p], lr=0.001)
        expected = torch.tensor([0.998])

        # Act
        p.grad = torch.tensor([0.1])
        adam.step()
        p.grad = torch.tensor([0.1])
        adam.step()

        # Assert
        self.assertTrue(torch.allclose(p, expected, atol=1e-4))

    def test_when_weight_decay_given_then_moments_are_computed_from_grad_plus_decay_times_p(
        self,
    ):
        # Arrange
        p = torch.tensor([1.0], requires_grad=True)
        p.grad = torch.tensor([0.1])
        weight_decay = 0.1
        adam = Adam([p], lr=0.001, weight_decay=weight_decay)
        effective_grad = p.grad + weight_decay * p
        expected_m = (1 - adam.beta1) * effective_grad
        expected_v = (1 - adam.beta2) * effective_grad**2

        # Act
        adam.step()

        # Assert
        self.assertTrue(torch.allclose(adam.m[0], expected_m))
        self.assertTrue(torch.allclose(adam.v[0], expected_v))


class TestAdamWStep(unittest.TestCase):
    def test_when_weight_decay_is_zero_then_matches_adams_update_exactly(self):
        # Arrange
        p_adam = torch.tensor([1.0], requires_grad=True)
        p_adamw = torch.tensor([1.0], requires_grad=True)
        lr = 0.001
        adam = Adam([p_adam], lr=lr, weight_decay=0.0)
        adamw = AdamW([p_adamw], lr=lr, weight_decay=0.0)

        # Act
        for _ in range(2):
            p_adam.grad = torch.tensor([0.1])
            adam.step()
            p_adamw.grad = torch.tensor([0.1])
            adamw.step()

        # Assert
        self.assertTrue(torch.allclose(p_adam, p_adamw, atol=1e-8))

    def test_when_weight_decay_given_then_differs_from_adams_l2_style_decay(self):
        # Arrange
        p_adam = torch.tensor([1.0], requires_grad=True)
        p_adamw = torch.tensor([1.0], requires_grad=True)
        lr = 0.001
        weight_decay = 0.1
        adam = Adam([p_adam], lr=lr, weight_decay=weight_decay)
        adamw = AdamW([p_adamw], lr=lr, weight_decay=weight_decay)
        expected_adam = torch.tensor([0.99800001])
        expected_adamw = torch.tensor([0.99780031])

        # Act
        for _ in range(2):
            p_adam.grad = torch.tensor([0.1])
            adam.step()
            p_adamw.grad = torch.tensor([0.1])
            adamw.step()

        # Assert
        self.assertTrue(torch.allclose(p_adam, expected_adam, atol=1e-6))
        self.assertTrue(torch.allclose(p_adamw, expected_adamw, atol=1e-6))
        self.assertFalse(torch.allclose(p_adam, p_adamw, atol=1e-6))
