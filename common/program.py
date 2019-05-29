import tensorflow as tf


class Program(object):
    def __init__(self):
        """
        Program
        """
        self.rules = {}

    def add_rule(self, rule):
        """ Adds a rule into the program

        :param rule: A Rule object
        """
        pass

    def forward_pass(self, model):
        """ A fuzzy consequence operator

            :param model: The input to the consequence operator
            :return: A valuation of the entailed atoms given the program and the model
            """
        pass

    def supported_loss(self, model):
        """ Fuzzy measure of supported model semantics

        :param model: input model taking values in the interval [0,1]
        :return: A measure of how supported the model is under the program
        """
        return tf.reduce_sum(tf.square(model - self.forward_pass(model)))
