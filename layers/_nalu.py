import tensorflow as tf

def NAC_init(in_dims, units, initializer):
    with tf.name_scope('nac_initializer'):
        w_shape = [in_dims, units]
        W_hat = tf.Variable(initializer(shape=w_shape), name='W_hat')
        M_hat = tf.Variable(initializer(shape=w_shape), name='M_hat')

        return tf.multiply(tf.nn.tanh(W_hat),
                           tf.nn.sigmoid(M_hat), name='W')

def NAC(in_tensor, units=None, initializer=None, W=None, activation=None):
    with tf.name_scope('nac_cell'):
        if W is None:
            W = NAC_init(in_tensor.get_shape().as_list()[-1],
                         units, initializer)

        if activation is None:
            return tf.matmul(in_tensor, W, name='out')
        return activation(tf.matmul(in_tensor, W, name='out'))

def NALU(in_tensor, units, initializer, epsilon=1e-10, activation=None):
    with tf.name_scope('nalu'):
        w_shape = [in_tensor.get_shape().as_list()[-1], units]
        W = NAC_init(w_shape[0],
                     units, initializer)

        with tf.name_scope('sigmoidal_gate'):
            G = tf.Variable(tf.glorot_normal_initializer()(shape=w_shape), name='G')
            g = tf.nn.sigmoid(tf.matmul(tf.abs(in_tensor), G), 'g')

        with tf.name_scope('nac_cells'):
            a = NAC(in_tensor, W=W)
            log_in = tf.log(tf.abs(in_tensor) + epsilon)
            m = tf.exp(NAC(log_in, W=W))

        if activation is None:
            return tf.add(tf.multiply(g, a),
                          tf.multiply(tf.subtract(1., g), m), 'y')
        return activation(tf.add(tf.multiply(g, a),
                          tf.multiply(tf.subtract(1., g), m), 'y'))
