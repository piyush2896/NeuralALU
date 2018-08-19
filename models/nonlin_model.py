import tensorflow as tf
from layers import NALU, NAC

initializer = tf.initializers.random_normal

def nonlin_nac_fn(features, labels, mode, params):
    nac1 = NAC(features['X'], units=8, initializer=initializer(0., 1.))
    nac2 = NAC(nac1, units=8, initializer=initializer(0., 1.))
    nac3 = NAC(nac2, units=8, initializer=initializer(0., 1.))
    logits = NAC(nac3, units=1, initializer=initializer(0., 1.))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y': logits}
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        loss = tf.losses.mean_squared_error(labels, logits)
        optimizer = tf.train.RMSPropOptimizer(1e-3)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
    return spec

def nonlin_nalu_fn(features, labels, mode, params):
    nalu1 = NALU(features['X'], units=8, initializer=initializer(0., 1.))
    nalu2 = NALU(nalu1, units=8, initializer=initializer(0., 1.))
    nalu3 = NALU(nalu2, units=8, initializer=initializer(0., 1.))
    logits = NALU(nalu3, units=1, initializer=initializer(0., 1.))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y': logits}
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        loss = tf.losses.mean_squared_error(labels, logits)
        optimizer = tf.train.RMSPropOptimizer(1e-3)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
    return spec
