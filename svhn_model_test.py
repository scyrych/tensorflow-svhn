import tensorflow as tf

slim = tf.contrib.slim

from svhn_model import conv_model


class ModetTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        height, width = 32, 32
        num_classes = 5
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 1))
            logits_1, logits_2, logits_3, logits_4, logits_5 = conv_model(inputs, num_classes)
            self.assertListEqual(logits_1.get_shape().as_list(),
                                 [batch_size, num_classes])

    def testEvaluation(self):
        batch_size = 2
        height, width = 32, 32
        num_classes = 5
        with self.test_session():
            eval_inputs = tf.random_uniform((batch_size, height, width, 3))
            logits_1, logits_2, logits_3, logits_4, logits_5 = conv_model(eval_inputs, num_classes, is_training=False)
            y_pred = tf.stack([logits_1, logits_2, logits_3, logits_4, logits_5])
            y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))
            print(y_pred_cls.get_shape().as_list()[0])
            self.assertListEqual(y_pred_cls.get_shape().as_list(),
                                 [batch_size, num_classes])
            self.assertListEqual([y_pred_cls.get_shape()[0]], [batch_size])

    def testForward(self):
        batch_size = 1
        height, width = 32, 32
        num_classes = 5
        with self.test_session() as sess:
            inputs = tf.random_uniform((batch_size, height, width, 1))
            logits_1, logits_2, logits_3, logits_4, logits_5 = conv_model(inputs, num_classes, is_training=True)
            y_pred = tf.nn.softmax(tf.stack([logits_1, logits_2, logits_3, logits_4, logits_5]))
            y_pred_cls = tf.transpose(tf.argmax(y_pred, dimension=2))
            prediction = tf.reduce_min(tf.cast(tf.equal(y_pred_cls, [0, 2, 1, 3, 2]), tf.float32), 1)
            accuracy = tf.reduce_mean(prediction) * 100.0
            sess.run(tf.global_variables_initializer())
            output = sess.run(accuracy)

            print(output)
            self.assertTrue(output.any())

if __name__ == '__main__':
    tf.test.main()
