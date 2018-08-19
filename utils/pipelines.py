import tensorflow as tf

def input_fn(X, y, buffer_size, batch_size, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_train:
        dataset = dataset.shuffle(buffer_size).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2000).repeat(None if is_train else 1)

    X, y = dataset.make_one_shot_iterator().get_next()
    X = tf.cast(X, tf.float32)
    y = tf.cast(y, tf.float32)
    return {'X': X}, y
