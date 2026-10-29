# import unittest
#
# from dl_lib.engine.value import Value
#
#
# class TestInit(unittest.TestCase):
#
#     def test_when_constructed_with_data_then_stores_it(self):
#         # Arrange
#         expected = 2.0
#
#         # Act
#         actual = Value(2.0).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_without_label_then_label_defaults_to_empty_string(
#             self):
#         # Arrange
#         expected = ""
#
#         # Act
#         actual = Value(2.0).label
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_with_label_then_stores_it(self):
#         # Arrange
#         expected = "a"
#
#         # Act
#         actual = Value(2.0, label="a").label
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_constructed_then_grad_defaults_to_zero(self):
#         # Arrange
#         expected = 0.0
#
#         # Act
#         actual = Value(2.0).grad
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestRepr(unittest.TestCase):
#
#     def test_when_printed_then_shows_data(self):
#         # Arrange
#         value = Value(2.0)
#         expected = "Value(data=2.0)"
#
#         # Act
#         actual = repr(value)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_data_is_negative_then_shows_the_negative_sign(self):
#         # Arrange
#         value = Value(-3.0)
#         expected = "Value(data=-3.0)"
#
#         # Act
#         actual = repr(value)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestAdd(unittest.TestCase):
#
#     def test_when_two_values_added_then_data_is_sum(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         expected = -1.0
#
#         # Act
#         actual = (a + b).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_two_values_added_then_result_tracks_operands_and_op(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         expected_prev = {a, b}
#         expected_op = "+"
#
#         # Act
#         out = a + b
#
#         # Assert
#         self.assertEqual(out._prev, expected_prev)
#         self.assertEqual(out._op, expected_op)
#
#     def test_when_value_added_to_float_then_data_is_sum(self):
#         # Arrange
#         a = Value(2.0)
#         expected = 5.0
#
#         # Act
#         actual = (a + 3.0).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestRAdd(unittest.TestCase):
#
#     def test_when_float_added_to_value_then_data_is_sum(self):
#         # Arrange
#         a = Value(2.0)
#         expected = 5.0
#
#         # Act
#         actual = (3.0 + a).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestMul(unittest.TestCase):
#
#     def test_when_two_values_multiplied_then_data_is_product(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         expected = -6.0
#
#         # Act
#         actual = (a * b).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_two_values_multiplied_then_result_tracks_operands_and_op(
#             self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         expected_prev = {a, b}
#         expected_op = "*"
#
#         # Act
#         out = a * b
#
#         # Assert
#         self.assertEqual(out._prev, expected_prev)
#         self.assertEqual(out._op, expected_op)
#
#     def test_when_value_multiplied_by_float_then_data_is_product(self):
#         # Arrange
#         a = Value(2.0)
#         expected = 6.0
#
#         # Act
#         actual = (a * 3.0).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestRMul(unittest.TestCase):
#
#     def test_when_float_multiplied_by_value_then_data_is_product(self):
#         # Arrange
#         a = Value(2.0)
#         expected = 6.0
#
#         # Act
#         actual = (3.0 * a).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestTanh(unittest.TestCase):
#
#     def test_when_called_then_data_matches_hand_calculation(self):
#         # Arrange
#         value = Value(0.0)
#         expected = 0.0
#
#         # Act
#         actual = value.tanh().data
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_result_tracks_operand_and_op(self):
#         # Arrange
#         value = Value(0.0)
#         expected_prev = {value}
#         expected_op = "tanh"
#
#         # Act
#         out = value.tanh()
#
#         # Assert
#         self.assertEqual(out._prev, expected_prev)
#         self.assertEqual(out._op, expected_op)
#
#
# class TestBackward(unittest.TestCase):
#
#     def test_when_called_on_sum_of_distinct_values_then_each_operand_gets_gradient_one(
#         self, ):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = a + b
#         expected_a_grad = 1.0
#         expected_b_grad = 1.0
#
#         # Act
#         c.backward()
#
#         # Assert
#         self.assertEqual(a.grad, expected_a_grad)
#         self.assertEqual(b.grad, expected_b_grad)
#
#     def test_when_called_then_root_gradient_is_one(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = a + b
#         expected = 1.0
#
#         # Act
#         c.backward()
#
#         # Assert
#         self.assertEqual(c.grad, expected)
#
#     def test_when_called_on_product_of_distinct_values_then_each_operand_gets_the_others_data(
#         self, ):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = a * b
#         expected_a_grad = -3.0
#         expected_b_grad = 2.0
#
#         # Act
#         c.backward()
#
#         # Assert
#         self.assertEqual(a.grad, expected_a_grad)
#         self.assertEqual(b.grad, expected_b_grad)
#
#     def test_when_called_on_tanh_then_operand_gets_the_local_tanh_derivative(
#             self):
#         # Arrange
#         a = Value(0.0)
#         t = a.tanh()
#         expected = 1.0
#
#         # Act
#         t.backward()
#
#         # Assert
#         self.assertAlmostEqual(a.grad, expected)
#
#     def test_when_value_is_reused_then_its_gradient_accumulates_across_every_use(
#             self):
#         # Arrange
#         a = Value(3.0)
#         b = a + a
#         expected = 2.0
#
#         # Act
#         b.backward()
#
#         # Assert
#         self.assertEqual(a.grad, expected)
#
#
# class TestPow(unittest.TestCase):
#
#     def test_when_raised_to_float_power_then_data_matches_hand_calculation(
#             self):
#         # Arrange
#         a = Value(2.0)
#         expected = 8.0
#
#         # Act
#         actual = (a**3.0).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_backward_called_then_operand_gradient_matches_power_rule(
#             self):
#         # Arrange
#         a = Value(2.0)
#         out = a**3.0
#         expected = 3.0 * 2.0**2.0
#
#         # Act
#         out.backward()
#
#         # Assert
#         self.assertEqual(a.grad, expected)
#
#
# class TestTrueDiv(unittest.TestCase):
#
#     def test_when_value_divided_by_float_then_data_matches_hand_calculation(
#             self):
#         # Arrange
#         a = Value(6.0)
#         expected = 3.0
#
#         # Act
#         actual = (a / 2.0).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_value_divided_by_value_then_data_matches_hand_calculation(
#             self):
#         # Arrange
#         a = Value(6.0)
#         b = Value(2.0)
#         expected = 3.0
#
#         # Act
#         actual = (a / b).data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestExp(unittest.TestCase):
#
#     def test_when_called_then_data_matches_hand_calculation(self):
#         # Arrange
#         a = Value(0.0)
#         expected = 1.0
#
#         # Act
#         actual = a.exp().data
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_backward_called_then_operand_gradient_equals_its_own_result(
#             self):
#         # Arrange
#         a = Value(2.0)
#         out = a.exp()
#         expected = out.data
#
#         # Act
#         out.backward()
#
#         # Assert
#         self.assertAlmostEqual(a.grad, expected)
