# import unittest
import tensorflow as tf
from common.rule import Rule


class Test(tf.test.TestCase):
    def test_rule_construction(self):
        weight = tf.constant(1.0, dtype=tf.float32)
        head = tf.constant(1.0, dtype=tf.float32)
        body = tf.constant([True, False], dtype=tf.float32)
        rule = Rule(weight, head, body)

    def test_rule_consequence(self):
        weight = tf.constant(1.0, dtype=tf.float32)
        head = tf.constant(1.0, dtype=tf.float32)
        body = tf.constant([True, False], dtype=tf.bool)
        rule = Rule(weight, head, body)

        variables = tf.constant([0.2, 0.8], dtype=tf.float32)
        actual = rule.consequence(variables)
        expected = tf.constant(0.04, dtype=tf.float32)

        self.assertAllClose(actual, expected)