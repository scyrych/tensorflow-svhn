import os
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from svhn_data import get_datasets
from svhn_model import conv_model

CHECKPOINT_PATH = 'checkpoints/svhn_multi/'
LOG_DIR = 'logs/svhn_multi/'
BATCH_SIZE = 128
DROPOUT_RATIO = 0.9
NUM_ITERATIONS = 400


# noinspection PyBroadException
def _restore_checkpoint():
    try:
        print('Restoring last checkpoint.')

        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_PATH)
        saver.restore(session, save_path=last_checkpoint)
        print("Restored checkpoint from:", last_checkpoint)

    except:
        print('Failed to restore latest checkpoint - initialize variables')
        session.run(tf.global_variables_initializer())


def _feed_train_dict(current_step=0):
    offset = (current_step * BATCH_SIZE) % (y_train.shape[0] - BATCH_SIZE)

    xs, ys = x_train[offset:offset + BATCH_SIZE], y_train[offset:offset + BATCH_SIZE]

    return {x: xs, y_: ys, keep_prob: DROPOUT_RATIO, is_training: 1}


def _evaluate_batch(is_test, batch_size):
    summed_accuracy = 0.0

    num_images = y_test.shape[0] if is_test else y_val.shape[0]
    batches = num_images // batch_size + 1

    for index in range(batches):
        offset = index * batch_size

        if is_test:
            xs, ys = x_test[offset:offset + batch_size], y_test[offset:offset + batch_size]
        else:
            xs, ys = x_val[offset:offset + batch_size], y_val[offset:offset + batch_size]

        summed_accuracy += session.run(accuracy,
                                       feed_dict={x: xs, y_: ys,
                                                  keep_prob: DROPOUT_RATIO,
                                                  is_training: 0})

    return summed_accuracy / (0.0 + batches)


def _run_session(display_step):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', session.graph)
    start_time = time.time()

    for step in range(NUM_ITERATIONS):
        summary, i, opt = session.run([merged, global_step, optimizer], feed_dict=_feed_train_dict(step))
        train_writer.add_summary(summary, i)

        if (i % display_step == 0) or (step == NUM_ITERATIONS - 1):
            batch_acc, l = session.run([accuracy, loss], feed_dict=_feed_train_dict(step))
            print('Minibatch loss at step %d: %f' % (i, l))
            print("Minibatch accuracy at step %d: %.4f" % (i, batch_acc))

            valid_acc = _evaluate_batch(is_test=False, batch_size=512)
            print("Validation accuracy at step %s: %.4f" % (i, valid_acc))

    run_time = time.time() - start_time
    print("\nTraining time usage: " + str(timedelta(seconds=int(round(run_time)))))

    test_acc = _evaluate_batch(is_test=True, batch_size=512)
    print("Test accuracy: %.4f" % test_acc)

    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    saver.save(session, save_path=CHECKPOINT_PATH, global_step=global_step)
    print('Model saved in file: {}'.format(CHECKPOINT_PATH))


x_train, y_train, x_val, y_val, x_test, y_test = get_datasets()

_, img_height, img_width, num_channels = x_train.shape
num_digits, num_labels = y_train.shape[1], len(np.unique(y_train))

print('Training set', x_train.shape, y_train.shape)
print('Validation set', x_val.shape, y_val.shape)
print('Test set', x_test.shape, y_test.shape)

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32,
                       shape=(None, img_height, img_width, num_channels),
                       name='x')

    y_ = tf.placeholder(tf.int64,
                        shape=(None, num_digits),
                        name='y_')

with tf.name_scope('keep_prob'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob', keep_prob)

with tf.name_scope('is_training'):
    is_training = tf.placeholder(tf.float32)

logits_1, logits_2, logits_3, logits_4, logits_5 = conv_model(x, num_labels, keep_prob, is_training)

with tf.name_scope('loss'):
    loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_1, labels=y_[:, 0]))
    loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_2, labels=y_[:, 1]))
    loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_3, labels=y_[:, 2]))
    loss4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_4, labels=y_[:, 3]))
    loss5 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_5, labels=y_[:, 4]))

    loss = loss1 + loss2 + loss3 + loss4 + loss5
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(1e-3, global_step, 5000, 0.1, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.name_scope('accuracy'):
    y_pred = tf.nn.softmax(tf.stack([
        tf.nn.softmax(logits_1),
        tf.nn.softmax(logits_2),
        tf.nn.softmax(logits_3),
        tf.nn.softmax(logits_4),
        tf.nn.softmax(logits_5)]))

    y_pred_class = tf.transpose(tf.cast(tf.argmax(y_pred, dimension=2), tf.float32))
    prediction = tf.reduce_min(tf.cast(tf.equal(tf.cast(y_pred_class, tf.int32), tf.cast(y_, tf.int32)), tf.float32), 1)
    accuracy = tf.reduce_mean(prediction) * 100.0

    tf.summary.scalar('accuracy', accuracy)

session = tf.Session()
saver = tf.train.Saver()

_restore_checkpoint()

_run_session(display_step=200)
