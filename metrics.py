"""
Scoring functions

Code blocks taken from Toni Heittola's repository: http://tut-arg.github.io/sed_eval/

Implementation of the Metrics in the following paper:
Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen, 'Metrics for polyphonic sound event detection',
Applied Sciences, 6(6):162, 2016
"""

import keras.backend as K
import tensorflow as tf


def threshold_binarize(x, threshold=0.5):
    """ Thresholds tensor, making each element
        that is more than 0.5 equal to 1.
    """
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def f1_overall_framewise(y_true, y_pred):
    y_pred = threshold_binarize(y_pred)
    y_true = threshold_binarize(y_true)
    difference = ((2 * y_true) - y_pred)
    true_positive = tf.equal(difference, 1)
    true_positive = K.cast(true_positive, K.floatx())
    true_positive = K.sum(true_positive)
    n_ref = K.cast(K.sum(y_true), K.floatx())
    n_sys = K.cast(K.sum(y_pred), K.floatx())
    precision = true_positive / (n_sys + K.epsilon())
    recall = true_positive / (n_ref + K.epsilon())

    f1_score = (2.0 * precision * recall) / (precision + recall + K.epsilon())
    return f1_score


def er_overall_framewise(y_true, y_pred):
    y_pred = threshold_binarize(y_pred)
    y_pred = K.cast(y_pred, K.floatx())
    y_true = threshold_binarize(y_true)
    y_true = K.cast(y_true, K.floatx())
    false_positive = tf.math.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1))
    false_positive = K.cast(false_positive, K.floatx())
    false_positive = K.sum(false_positive, axis=2)
    false_negative = tf.math.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0))
    false_negative = K.cast(false_negative, K.floatx())
    false_negative = K.sum(false_negative, axis=2)
    substitutions = K.sum(tf.keras.backend.minimum(false_positive, false_negative))
    deletes = K.sum(tf.keras.backend.maximum(0.0, false_negative-false_positive))
    insertions = K.sum(tf.keras.backend.maximum(0.0, false_positive-false_negative))
    n_ref = K.sum(y_true)

    error_rate = (substitutions + deletes + insertions) / (n_ref + K.epsilon())
    return error_rate
