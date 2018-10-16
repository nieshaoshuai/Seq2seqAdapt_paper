# -*- coding: utf-8 -*-
"""

This implementation is based on the original paper of Gao Huang, Zhuang Liu, Kilian Q. Weinberger and Laurens van der Maaten.
Besides I took some influences by random implementations, especially of Zhuang Liu's Lua implementation.
# References
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- DenseNet - keras implemented[https://github.com/cmasch/densenet/]
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from normalization import batch_norm


def dropout(incoming, is_training, keep_prob=0.5):
    return tf.contrib.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def DenseNet(
        img_input=None,
        dense_blocks=3,
        dense_layers=[6, 12, 16],
        growth_rate=24,
        dropout_rate=None,
        bottleneck=True,
        compression=0.5,
        depth=32,
        is_training=True):
    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:

    """

    if compression <= 0.0 or compression > 1.0:
        raise Exception('Compression have to be a value between 0.0 and 1.0.')

    if len(dense_layers) != dense_blocks:
        raise AssertionError(
            'Number of dense blocks have to be same length to specified layers')

    nb_channels = growth_rate

    # print('Creating DenseNet %s' % __version__)
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')

    # Initial convolution layer
    x = conv2d(img_input, 2 * growth_rate, ks=3, s=1, padding="SAME", name="dense_init_conv")
    x = tf.nn.max_pool(x, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), padding='SAME',
                       name="dense_pool")
    print(img_input.shape)
    print(x.shape)
    # Building dense blocks
    for block in range(dense_blocks - 1):
        # Add dense block
        block_name = "dense_block_{}".format(str(block))
        with tf.variable_scope(block_name):
            x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate,
                                         dropout_rate,
                                         bottleneck, is_training)

            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, is_training)
            nb_channels = int(nb_channels * compression)

    with tf.variable_scope("sencond_dense_block"):
        x, nb_channels = dense_block(x, dense_layers[-1], nb_channels, growth_rate, dropout_rate,
                                     is_training)
        x = batch_norm(x, is_training, scope="bn_s")
        x = tf.nn.relu(x, name="relu_s")
        x = tf.nn.avg_pool(x, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), name="avg_pool_s",
                           padding="SAME")
    # Add last dense block without transition but for that with global average pooling
    with tf.variable_scope("last_dense_block"):
        x, nb_channels = dense_block(x, dense_layers[-1], nb_channels, growth_rate, dropout_rate,
                                     is_training)
        x = batch_norm(x, is_training=is_training, scope="bn_e")
        x = tf.nn.relu(x, name="relu_e")
        x = tf.nn.avg_pool(x, ksize=(1, 2, 1, 1), strides=(1, 2, 1, 1), name="avg_pool_e",
                           padding="SAME")
    x = tf.squeeze(x, axis=1)

    # x_shape = array_ops.shape(x)
    # print(x_shape)
    # sequence_shape = [x_shape[0], x.shape[1]*x.shape[2], x_shape[3]]
    # print(sequence_shape)
    # # sequence_shape = np.array(sequence_shape, dtype=np.int)
    # x = tf.reshape(x, shape=tf.shape(sequence_shape))
    # x = tf.squeeze(x,axis=)
    return x


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False,
                is_training=True):
    """
    Creates a dense block and concatenates inputs
    """

    x_list = [x]
    for i in range(nb_layers):
        with tf.variable_scope("conv_block_{}".format(str(i))):
            cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, is_training)
            print(x.shape)
            print(i, "cb.shape", cb.shape)
        x_list.append(cb)
        x = tf.concat(x_list, axis=-1)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(input, nb_channels, dropout_rate=None, bottleneck=False, is_training=True):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        x = batch_norm(input, is_training, scope="bn_b")
        x = tf.nn.relu(x, name="relu_b")
        x = conv2d(x, nb_channels * bottleneckWidth, ks=1, s=1, name="conv1d")
        # Dropout
        if dropout_rate:
            x = dropout(x, is_training=is_training, keep_prob=1 - dropout_rate)
    else:
        x = input

    print("bottleneck.shape", input.shape)
    print("bottleneck.shape", x.shape)
    # Standard (BN-ReLU-Conv)
    x = batch_norm(x, is_training, scope="bn_c")
    x = tf.nn.relu(x, name="relu_c")
    x = conv2d(x, nb_channels, ks=3, s=1, name="conv_c")
    print("conv2d.shape", x.shape)
    # Dropout
    if dropout_rate:
        x = dropout(x, is_training=is_training, keep_prob=1 - dropout_rate)

    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, is_training=True):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.
    """
    x = batch_norm(x, is_training, scope="trans")
    x = tf.nn.relu(x)
    x = conv2d(x, int(nb_channels * compression), ks=1, s=1, name="conv_trans")

    # Adding dropout
    if dropout_rate:
        x = dropout(x, is_training=is_training, keep_prob=1 - dropout_rate)
    x = tf.nn.avg_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME",
                       name="avg_pool_trans")
    return x


if __name__ == "__main__":
    input_img = tf.placeholder(shape=[None, 64, 256, 3], dtype=tf.float32)
    feature = DenseNet(img_input=input_img,
                       dense_blocks=3,
                       dense_layers=[6, 12, 16],
                       growth_rate=24,
                       dropout_rate=None,
                       bottleneck=True,
                       compression=0.5,
                       depth=32,
                       is_training=True)
    print(feature.shape)
