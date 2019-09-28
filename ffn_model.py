"""
Architecture for Few-Shot Regression (Sinusoid and Multimodal)
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
# from .BaseModel import BaseModel


class FeatureExtractor(object):
    def __init__(self, inputs):
        if FLAGS.datasource == 'sinusoid':
            self.n_units = [40, 40, 40]
        else:
            self.n_units = [128, 128, 128, 128, 128, 128]
        with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
            self.build_model(inputs)

    def build_model(self, inputs):
        running_output = inputs
        for i, size in enumerate(self.n_units[:-1]):
            running_output = tf.nn.relu(
                tf.layers.dense(running_output, size, name="layer_{}".format(i)))
            # Last layer without a ReLu
        running_output = tf.layers.dense(
            running_output, self.n_units[-1], name="layer_{}".format(i + 1))
        self.output = running_output  # shape = (meta_batch_size, num_shot_train, 40)


class WeightsGenerator(object):
    def __init__(self, inputs, hidden, attention_layers):
        if FLAGS.datasource == 'mnist':
            output_units = 128
        elif FLAGS.datasource == 'sinusoid':
             output_units = 40
        with tf.variable_scope("attention"):
            train_embed = inputs
            for i in np.arange(attention_layers):
                query = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="query_{}".format(i))
                key = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="key_{}".format(i))
                value = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="value_{}".format(i))
                train_embed, _ = self.attention(query, key, value)
                dense = tf.layers.dense(inputs=train_embed, units=hidden*2, activation=tf.nn.relu, name="ff_layer{}_dense0".format(i))
                train_embed += tf.layers.dense(inputs=dense, units=hidden, activation=None, name="ff_layer{}_dense1".format(i))
                train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

            train_embed = tf.layers.dense(
                inputs=train_embed,
                units=output_units,
                activation=None,
            )
            self.final_weights = tf.reduce_mean(train_embed, axis=1, keep_dims=True)
            print(self.final_weights, 'final_weights\n')

    def attention(self, query, key, value):
        dotp = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[-1], tf.float32) ** 0.5)
        attention_weights = tf.nn.softmax(dotp)
        weighted_sum = tf.matmul(attention_weights, value)
        output = weighted_sum + query
        output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
        return output, attention_weights

    def mlp(self, input, output_sizes, name):
        """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

        Args:
          input: input tensor of shape [B,n,d_in].
          output_sizes: An iterable containing the output sizes of the MLP as defined
              in `basic.Linear`.
          variable_scope: String giving the name of the variable scope. If this is set
              to be the same as a previously defined MLP, then the weights are reused.

        Returns:
          tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
        """
        # Get the shapes of the input and reshape to parallelise across observations
        # batch_size, _, filter_size = input.shape.as_list()
        # output = tf.reshape(input, (-1, filter_size))
        # output.set_shape((None, filter_size))
        output = input
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i, size in enumerate(output_sizes[:-1]):
                output = tf.nn.relu(
                    tf.layers.dense(output, size, name="layer_{}".format(i), use_bias=False))
            # Last layer without a ReLu
            output = tf.layers.dense(
                output, output_sizes[-1], name="layer_{}".format(i + 1), use_bias=False)
        return output


class FFN():

    def __init__(self, name, num_train_samples=10, num_test_samples=10, l1_penalty=0.001, l2_penalty=0.001):
        # super(FFN, self).__init__()
        self.name = name
        # Attention parameters
        self.attention_layers = 3
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.hidden = 64
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.build_model(num_train_samples, num_test_samples)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)

    def build_model(self, num_train_samples=10, num_test_samples=10):
        self.train_inputs = tf.placeholder(
            shape=(None, None, 1 if FLAGS.datasource == 'sinusoid' else 2),
            dtype=tf.float32,
            name="train_inputs",
        )
        self.train_labels = tf.placeholder(
            shape=(None, None, 1),
            dtype=tf.float32,
            name="train_labels",
        )
        self.test_inputs = tf.placeholder(
            shape=(None, None, 1 if FLAGS.datasource == 'sinusoid' else 2),
            dtype=tf.float32,
            name="test_inputs"
        )
        self.test_labels = tf.placeholder(
            shape=(None, None, 1),
            dtype=tf.float32,
            name="test_labels",
        )

        # Extract training features
        self.train_feature_extractor = FeatureExtractor(self.train_inputs)

        t_input = tf.concat([self.train_feature_extractor.output, self.train_labels], axis=-1)

        weights_generator_train = WeightsGenerator(t_input, self.hidden, self.attention_layers)
        self.train_final_weights = weights_generator_train.final_weights

        # Extract test features
        test_feature_extractor = FeatureExtractor(self.test_inputs)
        self.test_features = test_feature_extractor.output

        self.predictions = tf.matmul(self.test_features, self.train_final_weights, transpose_b=True)
        self.penalty_loss = self.l1_penalty * tf.norm(self.train_final_weights, ord=1) + \
                            self.l2_penalty * tf.norm(self.train_final_weights, ord=2)

        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels, [-1]),
                                                 predictions=tf.reshape(self.predictions,
                                                                        [-1])) + self.penalty_loss

        self.plain_loss = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels, [-1]),
                                                       predictions=tf.reshape(self.predictions, [-1]))


