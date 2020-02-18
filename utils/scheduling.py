import tensorflow as tf
import numpy as np


def linear_function(x, x0, x1, y0, y1):
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m * x + b


def piecewise_linear(step, X, Y):
    """
    Piecewise linear function.
    
    :param step: current step.
    :param X: list of breakpoints
    :param Y: list of values at breakpoints
    :return: value of piecewise linear function with values Y_i at step X_i
    """
    assert len(X) == len(Y)
    X = np.array(X)
    if step < X[0]:
        return Y[0]
    idx = np.where(step >= X)[0][-1]
    if idx == (len(Y) - 1):
        return Y[-1]
    else:
        return linear_function(step, X[idx], X[idx + 1], Y[idx], Y[idx + 1])


def piecewise_linear_schedule(step, schedule):
    x_schedule = schedule[:, 0]
    y_schedule = schedule[:, 1]
    value = piecewise_linear(step, x_schedule, y_schedule)
    return tf.cast(value, tf.float32)


def dropout_schedule(step, schedule):
    schedule = np.array(schedule)
    dout = piecewise_linear_schedule(step, schedule)
    return tf.cast(dout, tf.float32)


def learning_rate_schedule(step, schedule):
    schedule = np.array(schedule)
    lr = piecewise_linear_schedule(step, schedule)
    return tf.cast(lr, tf.float32)