from program import Program
from forward_pass import RuleObj
import tensorflow as tf


rule_weights = tf.Variable([0.9, 1.0, 1.0], dtype=tf.float32)
rule_negs = tf.constant([[False, True]], dtype=tf.bool)
rule_bodies = tf.constant([[[0], [1], [1], [2]]], dtype=tf.int64)

rule_obj = RuleObj(rule_weights, rule_negs, rule_bodies)

program = Program(rule_obj)

model = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32)

gs = program.gradients(model)
print(gs)