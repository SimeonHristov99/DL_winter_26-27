# import unittest
#
# from dl_lib.engine import graph
# from dl_lib.engine.value import Value
#
#
# class TestTrace(unittest.TestCase):
#
#     def test_when_called_on_leaf_then_returns_only_that_node_and_no_edges(
#             self):
#         # Arrange
#         a = Value(2.0)
#         expected_nodes = {a}
#         expected_edges = set()
#
#         # Act
#         actual_nodes, actual_edges = graph.trace(a)
#
#         # Assert
#         self.assertEqual(actual_nodes, expected_nodes)
#         self.assertEqual(actual_edges, expected_edges)
#
#     def test_when_called_on_expression_then_returns_every_node_and_edge(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = Value(10.0)
#         e = a * b
#         d = e + c
#         expected_nodes = {a, b, c, d, e}
#         expected_edges = {(a, e), (b, e), (e, d), (c, d)}
#
#         # Act
#         actual_nodes, actual_edges = graph.trace(d)
#
#         # Assert
#         self.assertEqual(actual_nodes, expected_nodes)
#         self.assertEqual(actual_edges, expected_edges)
#
#     def test_when_a_value_feeds_two_parents_then_it_is_visited_only_once(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = a + b
#         d = a * b
#         e = c + d
#         expected_nodes = {a, b, c, d, e}
#         expected_edges = {(a, c), (b, c), (a, d), (b, d), (c, e), (d, e)}
#
#         # Act
#         actual_nodes, actual_edges = graph.trace(e)
#
#         # Assert
#         self.assertEqual(actual_nodes, expected_nodes)
#         self.assertEqual(actual_edges, expected_edges)
#
#
# class TestDrawDot(unittest.TestCase):
#
#     def test_when_called_on_expression_then_renders_a_record_node_per_value_and_an_op_node_per_operation(
#             self):
#         # Arrange
#         a = Value(2.0, label="a")
#         b = Value(-3.0, label="b")
#         c = a + b
#         c.label = "c"
#         expected_a_record = "{ a | data: 2.0000 | grad: 0.0000 }"
#         expected_b_record = "{ b | data: -3.0000 | grad: 0.0000 }"
#         expected_c_record = "{ c | data: -1.0000 | grad: 0.0000 }"
#         expected_op_count = 1
#
#         # Act
#         dot = graph.draw_dot(c)
#         source = dot.source
#
#         # Assert
#         self.assertIn(expected_a_record, source)
#         self.assertIn(expected_b_record, source)
#         self.assertIn(expected_c_record, source)
#         self.assertEqual(source.count('label="+"'), expected_op_count)
#
#     def test_when_called_on_leaf_then_renders_only_one_record_node_and_no_op_node(
#             self):
#         # Arrange
#         a = Value(2.0, label="a")
#         expected_a_record = "{ a | data: 2.0000 | grad: 0.0000 }"
#         expected_node_count = 1
#
#         # Act
#         dot = graph.draw_dot(a)
#         source = dot.source
#
#         # Assert
#         self.assertIn(expected_a_record, source)
#         self.assertEqual(source.count("shape=record"), expected_node_count)
#         self.assertEqual(source.count("label="), expected_node_count)
#
#
# class TestTopSort(unittest.TestCase):
#
#     def test_when_called_on_leaf_then_returns_only_that_node(self):
#         # Arrange
#         a = Value(2.0)
#         expected = [a]
#
#         # Act
#         actual = graph.top_sort(a)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_on_expression_then_every_operand_precedes_its_result(
#             self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = Value(10.0)
#         e = a * b
#         d = e + c
#
#         # Act
#         order = graph.top_sort(d)
#
#         # Assert
#         self.assertLess(order.index(a), order.index(e))
#         self.assertLess(order.index(b), order.index(e))
#         self.assertLess(order.index(e), order.index(d))
#         self.assertLess(order.index(c), order.index(d))
#         self.assertEqual(order[-1], d)
#
#     def test_when_a_value_feeds_two_parents_then_it_appears_only_once(self):
#         # Arrange
#         a = Value(2.0)
#         b = Value(-3.0)
#         c = a + b
#         d = a * b
#         e = c + d
#
#         # Act
#         order = graph.top_sort(e)
#
#         # Assert
#         expected_count = 1
#         self.assertEqual(order.count(a), expected_count)
#         self.assertEqual(order.count(b), expected_count)
#         self.assertLess(order.index(a), order.index(c))
#         self.assertLess(order.index(b), order.index(c))
#         self.assertLess(order.index(a), order.index(d))
#         self.assertLess(order.index(b), order.index(d))
#         self.assertEqual(order[-1], e)
