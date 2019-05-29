import tensorflow as tf


@tf.function
def consequences(program, model):
    """ A fuzzy consequence operator

    :param program: A Program object
    :param model: The input to the consequence operator
    :return: A valuation of the entailed atoms given the program and the model
    """
    pass


@tf.function
def supported_loss(program, model):
    pass