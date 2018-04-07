import tensorflow.contrib.slim as slim
import tensorflow as tf


def conv_model(images, num_labels, dropout_keep_prob=0.5, is_training=0):
    # Block 1
    net = slim.conv2d(images, 32, [5, 5], scope='conv1_1')
    net = slim.conv2d(net, 32, [5, 5], scope='conv1_2')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

    # Block 2
    net = slim.conv2d(net, 64, [5, 5], scope='conv2_1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2_2')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

    # Block 3
    net = slim.conv2d(net, 128, [5, 5], scope='conv3_1')
    net = slim.conv2d(net, 128, [5, 5], scope='conv3_2')
    net = slim.conv2d(net, 128, [5, 5], scope='conv3_3')
    net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

    # Fully-connected 1
    net = slim.flatten(net)
    net = slim.fully_connected(net, 256, scope='fc4')

    # if is_training > 0:
    #     net = slim.dropout(net, dropout_keep_prob, scope='dropout4')

    # Fully-connected 2
    net = slim.fully_connected(net, 256, scope='fc5')
    # Softmax layers
    logits_1 = slim.fully_connected(net, num_labels, activation_fn=None,
                                    scope='softmax1')
    logits_2 = slim.fully_connected(net, num_labels, activation_fn=None,
                                    scope='softmax2')
    logits_3 = slim.fully_connected(net, num_labels, activation_fn=None,
                                    scope='softmax3')
    logits_4 = slim.fully_connected(net, num_labels, activation_fn=None,
                                    scope='softmax4')
    logits_5 = slim.fully_connected(net, num_labels, activation_fn=None,
                                    scope='softmax5')

    return logits_1, logits_2, logits_3, logits_4, logits_5
