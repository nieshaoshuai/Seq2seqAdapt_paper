import tensorflow as tf
import tensorflow.contrib as tf_contrib

def batch_norm(x, is_training=False, scope='batch_norm', set_updates_collections = False):
    '''
     when training, the moving_mean and moving_variance need to be updated.
     By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
     so they need to be added as a dependency to the train_op.

     For example:

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)

     One can set updates_collections=None to force the updates in place,
     but that can have a speed penalty, especially in distributed settings
    '''
    if set_updates_collections:
        return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, renorm=True, updates_collections=None,
                                        is_training=is_training, scope=scope)
    else:
        return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, renorm=True,
                                        is_training=is_training, scope=scope)

    # return tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-05, center=True, scale=True, renorm=True, training=is_training, name=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def group_norm(x, G=32, eps=1e-5, scope='group_norm') :
    with tf.variable_scope(scope) :
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x