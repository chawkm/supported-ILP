import tensorflow as tf
import numpy as np

def preprocess_rules_to_tf(ground_indexes, consequences):
    """ Prepare grounding for training

    :param ground_indexes: key value map of ground instances
    :param grounded_rules: list of grounded consequences [(w, body, negs)]
    :return: tf.ragged.constants for weights, bodies, and negations
    """
    N = len(ground_indexes)
    ground_indexes["_truth"] = N

    data_weights = []
    data_bodies = []
    data_negs = []

    max_rule_len = max(len(body[1]) for cons in consequences for body in cons)

    for cons in consequences:
        unzipped_cons = zip(*cons)
        weights = unzipped_cons[0]
        bodies = unzipped_cons[1]
        negs = unzipped_cons[2]
        for b, n in zip(bodies, negs):
            while len(b) < max_rule_len:
                # pad the body to match max length
                b.append(N)
                n.append(False)
        data_weights.append(np.array(weights).tolist())
        data_bodies.append(np.expand_dims(np.array(bodies), 2).tolist())
        data_negs.append(np.array(negs).tolist())

    print(data_weights, data_bodies, data_negs)
    return tf.ragged.constant(data_weights, ragged_rank=1), \
           tf.ragged.constant(data_bodies, ragged_rank=1), \
           tf.ragged.constant(data_negs, ragged_rank=1)

