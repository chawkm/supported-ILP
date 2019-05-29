import tensorflow as tf


class Rule(object):
    def __init__(self, weight, head, body):
        """
        Rule object
        """
        self.weight = weight
        self.head = head
        self.body = body

    @tf.function
    def consequence(self, variables):
        """ Returns fuzzy conjunction of body variables

        :param variables:
        :return:
        """
        return tf.reduce_prod(tf.where(self.body, variables, 1 - variables))


