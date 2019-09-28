import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
from tensorflow.python.platform import flags
import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS
flags.DEFINE_string('datasource', 'mnist', 'mnist or sinusoid (miniimagenet WIP)')
flags.DEFINE_integer('num_shot_train', 50,
                     'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('few_k_shot', 50,
                     'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('num_shot_test', 50, 'Number of test samples per class per task')
flags.DEFINE_integer('seed', 100, 'Set seed')
flags.DEFINE_bool('anp', False, '--')
flags.DEFINE_bool('training', True, '--')

# Training parameters
flags.DEFINE_integer('epochs', 500, 'Number of metatraining iterations')
flags.DEFINE_integer('batch_size', 80, 'Batchsize for metatraining')
flags.DEFINE_float('lr', 5e-4, 'Meta learning rate')
flags.DEFINE_string('savepath', 'saved_model/', 'Path to save or load models')
flags.DEFINE_string('gpu', '0', 'id of the gpu to use in the local machine')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRINT_INTERVAL = 200
TEST_PRINT_INTERVAL = 500
random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)
if FLAGS.datasource == 'mnist':
    num_tasks = 60000
    num_tasks_test = 10000
else:
    num_tasks = 240000
    num_tasks_test = 100

tf.logging.set_verbosity(tf.logging.ERROR)


# utility methods
def batch_mlp(input, output_sizes, variable_scope):
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
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(
                tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(
            output, output_sizes[-1], name="layer_{}".format(i + 1))
    # Bring back into original shape
    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))
    return output


class DeterministicEncoder(object):
  """The Deterministic Encoder."""

  def __init__(self, output_sizes, attention):
    """(A)NP deterministic encoder.

    Args:
      output_sizes: An iterable containing the output sizes of the encoding MLP.
      attention: The attention module.
    """
    self._output_sizes = output_sizes
    self._attention = attention

  def __call__(self, context_x, context_y, target_x):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
          task this corresponds to the x-values.
      context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
          task this corresponds to the y-values.
      target_x: Tensor of shape [B,target_observations,d_x].
          For this 1D regression task this corresponds to the x-values.

    Returns:
      The encoded representation. Tensor of shape [B,target_observations,d]
    """

    # Concatenate x and y along the filter axes, [B, observations, 2]
    encoder_input = tf.concat([context_x, context_y], axis=-1)

    # Pass final axis through MLP
    hidden = batch_mlp(encoder_input, self._output_sizes,
                       "deterministic_encoder")

    # Apply attention
    with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
        hidden = self._attention(context_x, target_x, hidden)

    return hidden


class LatentEncoder(object):
    """The Latent Encoder."""

    def __init__(self, output_sizes, num_latents):
        """(A)NP latent encoder.

        Args:
          output_sizes: An iterable containing the output sizes of the encoding MLP.
          num_latents: The latent dimensionality.
        """
        self._output_sizes = output_sizes
        self._num_latents = num_latents

    def __call__(self, x, y):
        """Encodes the inputs into one representation.

        Args:
          x: Tensor of shape [B,observations,d_x]. For this 1D regression
              task this corresponds to the x-values.
          y: Tensor of shape [B,observations,d_y]. For this 1D regression
              task this corresponds to the y-values.

        Returns:
          A normal distribution over tensors of shape [B, num_latents]
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes, "latent_encoder")

        # Aggregator: take the mean over all points
        hidden = tf.reduce_mean(hidden, axis=1)

        # Have further MLP layers that map to the parameters of the Gaussian latent
        with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
            # First apply intermediate relu layer
            hidden = tf.nn.relu(
                tf.layers.dense(hidden,
                                (self._output_sizes[-1] + self._num_latents) / 2,
                                name="penultimate_layer"))
            # Then apply further linear layers to output latent mu and log sigma
            mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
            log_sigma = tf.layers.dense(hidden, self._num_latents, name="std_layer")

        # Compute sigma
        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

        return tf.contrib.distributions.Normal(loc=mu, scale=sigma)


class Decoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        """(A)NP decoder.

        Args:
          output_sizes: An iterable containing the output sizes of the decoder MLP
              as defined in `basic.Linear`.
        """
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x):
        """Decodes the individual targets.

        Args:
          representation: The representation of the context for target predictions.
              Tensor of shape [B,target_observations,?].
          target_x: The x locations for the target query.
              Tensor of shape [B,target_observations,d_x].

        Returns:
          dist: A multivariate Gaussian over the target points. A distribution over
              tensors of shape [B,target_observations,d_y].
          mu: The mean of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x].
          sigma: The standard deviation of the multivariate Gaussian.
              Tensor of shape [B,target_observations,d_x].
        """
        # concatenate target_x and representation
        hidden = tf.concat([representation, target_x], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(hidden, self._output_sizes, "decoder")

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tf.contrib.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=sigma)

        return dist, mu, sigma


class LatentModel(object):
    """The (A)NP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                 decoder_output_sizes, use_deterministic_path=True,
                 deterministic_encoder_output_sizes=None, attention=None):
        """Initialises the model.

        Args:
          latent_encoder_output_sizes: An iterable containing the sizes of hidden
              layers of the latent encoder.
          num_latents: The latent dimensionality.
          decoder_output_sizes: An iterable containing the sizes of hidden layers of
              the decoder. The last element should correspond to d_y * 2
              (it encodes both mean and variance concatenated)
          use_deterministic_path: a boolean that indicates whether the deterministic
              encoder is used or not.
          deterministic_encoder_output_sizes: An iterable containing the sizes of
              hidden layers of the deterministic encoder. The last one is the size
              of the deterministic representation r.
          attention: The attention module used in the deterministic encoder.
              Only relevant when use_deterministic_path=True.
        """
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                             num_latents)
        self._decoder = Decoder(decoder_output_sizes)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
                deterministic_encoder_output_sizes, attention)

    def __call__(self, query, num_targets, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
          query: Array containing ((context_x, context_y), target_x) where:
              context_x: Tensor of shape [B,num_contexts,d_x].
                  Contains the x values of the context points.
              context_y: Tensor of shape [B,num_contexts,d_y].
                  Contains the y values of the context points.
              target_x: Tensor of shape [B,num_targets,d_x].
                  Contains the x values of the target points.
          num_targets: Number of target points.
          target_y: The ground truth y values of the target y.
              Tensor of shape [B,num_targets,d_y].

        Returns:
          log_p: The log_probability of the target_y given the predicted
              distribution. Tensor of shape [B,num_targets].
          mu: The mean of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
          sigma: The variance of the predicted distribution.
              Tensor of shape [B,num_targets,d_y].
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        prior = self._latent_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            latent_rep = prior.sample()
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.sample()
        latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                             [1, num_targets, 1])
        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y,
                                                            target_x)
            representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
        else:
            representation = latent_rep

        dist, mu, sigma = self._decoder(representation, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self._latent_encoder(target_x, target_y)
            kl = tf.reduce_sum(
                tf.contrib.distributions.kl_divergence(posterior, prior),
                axis=-1, keepdims=True)
            kl = tf.tile(kl, [1, num_targets])
            loss = - tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      v: values. tensor of shape [B,n,d_v].

    Returns:
      tensor of shape [B,m,d_v].
    """
    total_points = tf.shape(q)[1]
    rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B,1,d_v]
    rep = tf.tile(rep, [1, total_points, 1])
    return rep


def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
      q: queries. tensor of shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      scale: float that scales the L1 distance.
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
    q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
    unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = lambda x: 1 + tf.tanh(x)
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      normalise: Boolean that determines whether weights sum to 1.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = tf.shape(q)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
    if normalise:
        weight_fn = tf.nn.softmax
    else:
        weight_fn = tf.sigmoid
    weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
    return rep


def multihead_attention(q, k, v, num_heads=8):
    """Computes multi-head attention.

    Args:
      q: queries. tensor of  shape [B,m,d_k].
      k: keys. tensor of shape [B,n,d_k].
      v: values. tensor of shape [B,n,d_v].
      num_heads: number of heads. Should divide d_v.

    Returns:
      tensor of shape [B,m,d_v].
    """
    d_k = q.get_shape().as_list()[-1]
    d_v = v.get_shape().as_list()[-1]
    head_size = d_v / num_heads
    key_initializer = tf.random_normal_initializer(stddev=d_k ** -0.5)
    value_initializer = tf.random_normal_initializer(stddev=d_v ** -0.5)
    rep = tf.constant(0.0)
    for h in range(num_heads):
        o = dot_product_attention(
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wq%d' % h, use_bias=False, padding='VALID')(q),
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wk%d' % h, use_bias=False, padding='VALID')(k),
            tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                             name='wv%d' % h, use_bias=False, padding='VALID')(v),
            normalise=True)
        rep += tf.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                                name='wo%d' % h, use_bias=False, padding='VALID')(o)
    return rep


class Attention(object):
    """The Attention module."""

    def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
                 num_heads=8):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pair
        to output an aggregated representation of the context data.
        Args:
          rep: transformation to apply to contexts before computing attention.
              One of: ['identity','mlp'].
          output_sizes: list of number of hidden units per layer of mlp.
              Used only if rep == 'mlp'.
          att_type: type of attention. One of the following:
              ['uniform','laplace','dot_product','multihead']
          scale: scale of attention.
          normalise: Boolean determining whether to:
              1. apply softmax to weights so that they sum to 1 across context pts or
              2. apply custom transformation to have weights in [0,1].
          num_heads: number of heads for multihead.
        """
        self._rep = rep
        self._output_sizes = output_sizes
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        if self._type == 'multihead':
            self._num_heads = num_heads

    def __call__(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

        Args:
          x1: tensor of shape [B,n1,d_x].
          x2: tensor of shape [B,n2,d_x].
          r: tensor of shape [B,n1,d].

        Returns:
          tensor of shape [B,n2,d]

        Raises:
          NameError: The argument for rep/type was invalid.
        """
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            k = batch_mlp(x1, self._output_sizes, "attention")
            q = batch_mlp(x2, self._output_sizes, "attention")
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == 'uniform':
            rep = uniform_attention(q, r)
        elif self._type == 'laplace':
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == 'multihead':
            rep = multihead_attention(q, k, r, self._num_heads)
        else:
            raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                             ",'multihead']"))

        return rep


def NP_training():
    TRAINING_ITERATIONS = 0  # @param {type:"number"}
    MAX_CONTEXT_POINTS = FLAGS.num_shot_train  # @param {type:"number"}
    PLOT_AFTER = 10000  # @param {type:"number"}
    MODEL_TYPE = 'NP'  # @param ['NP','ANP']
    ATTENTION_TYPE = 'uniform'  # @param ['uniform','laplace','dot_product','multihead']
    random_kernel_parameters = True  # @param {type:"boolean"}

    tf.reset_default_graph()
    # Train dataset

    HIDDEN_SIZE = 40 if FLAGS.datasource=='sin' else 128
    latent_encoder_output_sizes = [HIDDEN_SIZE] * 2 if FLAGS.datasource=='sin' else [HIDDEN_SIZE] * 4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes = [HIDDEN_SIZE] * 2 if FLAGS.datasource=='sin' else [HIDDEN_SIZE] * 4

    decoder_output_sizes = [HIDDEN_SIZE] * 2 + [2]
    use_deterministic_path = True

    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
        attention = Attention(rep='mlp', output_sizes=[HIDDEN_SIZE] * 2,
                              att_type=ATTENTION_TYPE)
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
        attention = Attention(rep='identity', output_sizes=None, att_type='uniform')
    else:
        raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # Define the model
    model = LatentModel(latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path,
                        deterministic_encoder_output_sizes, attention)
    support_x = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1 if FLAGS.datasource == 'sin' else 2),
        dtype=tf.float32,
        name="support_x",
    )
    support_y = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1),
        dtype=tf.float32,
        name="support_y",
    )
    query_x = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1 if FLAGS.datasource == 'sin' else 2),
        dtype=tf.float32,
        name="query_x"
    )
    query_y = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1),
        dtype=tf.float32,
        name="query_y",
    )
    query = ((support_x, support_y), query_x)

    # Define the loss
    mu, sigma, log_prob, _, loss = model(query, tf.shape(query_x)[1], query_y)
    mse = tf.losses.mean_squared_error(labels=query_y, predictions=mu, reduction=tf.losses.Reduction.NONE)
    mse_all = tf.reduce_mean(mse, 1)
    mse = tf.reduce_mean(mse)

    Batch = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.lr, global_step=Batch,
                                               decay_steps=1e5, decay_rate=0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss, global_step=Batch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=10)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('mse', mse)
    tf.summary.scalar('lr', learning_rate)
    mergerd_sum = tf.summary.merge_all()
    config_str = 'NP_'
    config_str += FLAGS.datasource + '_' + str(FLAGS.num_shot_train) + 'shot' + '_lr' + str(FLAGS.lr)
    print(config_str)
    save_path = FLAGS.savepath + config_str + '/'
    if not FLAGS.training:
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print('\nload pretrained model\n')
    train_writer = tf.summary.FileWriter(save_path + 'train_writer', sess.graph)

    import reg_data_generator
    if FLAGS.datasource == 'mnist':
        dataset_train = reg_data_generator.MnistNP(MAX_CONTEXT_POINTS, MAX_CONTEXT_POINTS)
        dataset_test = reg_data_generator.MnistNP(MAX_CONTEXT_POINTS, 784 - MAX_CONTEXT_POINTS, False)
    elif FLAGS.datasource == 'sin':
        dataset_train = reg_data_generator.SinusoidGenerator(True)
        dataset_test = reg_data_generator.SinusoidGenerator(False)

    num_iters_per_epoch = int(num_tasks / FLAGS.batch_size)
    # if not FLAGS.fine_tune:
    #     os.mkdir(save_path + 'val_img')
    if FLAGS.training:
        for e_idx in range(FLAGS.epochs):
            # for each batch tasks
            perm = np.arange(0, num_tasks) if FLAGS.datasource=='sin' else np.random.permutation(num_tasks)
            for b_idx in range(num_iters_per_epoch):
                # count iter
                index = perm[FLAGS.batch_size * b_idx:FLAGS.batch_size * (b_idx + 1)]
                if FLAGS.datasource == 'mnist':
                    train_x, train_y, valid_x, valid_y = dataset_train.generate_batch(indx=index)
                else:
                    train_x, train_y, valid_x, valid_y = dataset_train.generate_batch(indx=index)

                feed_dict = {support_x: train_x, support_y: train_y, query_x: valid_x, query_y: valid_y}

                outs = sess.run([loss, train_step, mergerd_sum, mse, Batch], feed_dict)
                train_writer.add_summary(outs[2], outs[4])

    else:
        test_mse_list = []
        perm_test = np.random.permutation(num_tasks_test)
        for iii in range(int(num_tasks_test / FLAGS.batch_size)):
            index = perm_test[FLAGS.batch_size * iii:FLAGS.batch_size * (iii + 1)]
            if FLAGS.datasource == 'mnist':
                train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(indx=index)
            else:
                train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(indx=index)

            feed_dict = {support_x: train_x, support_y: train_y, query_x: valid_x, query_y: valid_y}
            outs = sess.run([mse], feed_dict)
            test_mse_list.append(outs[0])
        testing_mse = np.mean(test_mse_list)
        print('testing mse is ', testing_mse)


def ANP_training():
    TRAINING_ITERATIONS = 100000  # @param {type:"number"}
    MAX_CONTEXT_POINTS = FLAGS.num_shot_train  # @param {type:"number"}
    PLOT_AFTER = 10000  # @param {type:"number"}
    MODEL_TYPE = 'ANP'  # @param ['NP','ANP']
    ATTENTION_TYPE = 'multihead'  # @param ['uniform','laplace','dot_product','multihead']
    random_kernel_parameters = True  # @param {type:"boolean"}

    tf.reset_default_graph()

    HIDDEN_SIZE = 40 if FLAGS.datasource == 'sin' else 128
    latent_encoder_output_sizes = [HIDDEN_SIZE] * 2 if FLAGS.datasource == 'sin' else [HIDDEN_SIZE] * 4
    num_latents = HIDDEN_SIZE
    deterministic_encoder_output_sizes = [HIDDEN_SIZE] * 2 if FLAGS.datasource == 'sin' else [HIDDEN_SIZE] * 4
    decoder_output_sizes = [HIDDEN_SIZE] * 2 + [2]
    use_deterministic_path = True

    # ANP with multihead attention
    if MODEL_TYPE == 'ANP':
        attention = Attention(rep='mlp', output_sizes=[HIDDEN_SIZE] * 2,
                              att_type='multihead')
    # NP - equivalent to uniform attention
    elif MODEL_TYPE == 'NP':
        attention = Attention(rep='identity', output_sizes=None, att_type='uniform')
    else:
        raise NameError("MODEL_TYPE not among ['ANP,'NP']")

    # Define the model
    model = LatentModel(latent_encoder_output_sizes, num_latents,
                        decoder_output_sizes, use_deterministic_path,
                        deterministic_encoder_output_sizes, attention)

    support_x = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1 if FLAGS.datasource == 'sin' else 2),
        dtype=tf.float32,
        name="support_x",
    )
    support_y = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1),
        dtype=tf.float32,
        name="support_y",
    )
    query_x = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1 if FLAGS.datasource == 'sin' else 2),
        dtype=tf.float32,
        name="query_x"
    )
    query_y = tf.placeholder(
        shape=(FLAGS.batch_size, None, 1),
        dtype=tf.float32,
        name="query_y",
    )
    query = ((support_x, support_y), query_x)

    # Define the loss
    mu, sigma, log_prob, _, loss = model(query, tf.shape(query_x)[1], query_y)
    mse = tf.losses.mean_squared_error(labels=query_y, predictions=mu, reduction=tf.losses.Reduction.NONE)
    mse = tf.reduce_mean(mse)

    Batch = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.lr, global_step=Batch,
                                               decay_steps=1e5, decay_rate=0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss, global_step=Batch)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=10)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('mse', mse)
    tf.summary.scalar('lr', learning_rate)
    mergerd_sum = tf.summary.merge_all()
    config_str = 'ANP_'
    config_str += FLAGS.datasource + '_' + str(FLAGS.num_shot_train) + 'shot' + '_lr' + str(FLAGS.lr)
    print(config_str)
    save_path = FLAGS.savepath + config_str + '/'
    if not FLAGS.training:
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print('\nload pretrained model\n')
    train_writer = tf.summary.FileWriter(save_path + 'train_writer', sess.graph)

    import reg_data_generator
    if FLAGS.datasource == 'mnist':
        dataset_train = reg_data_generator.MnistNP(MAX_CONTEXT_POINTS, MAX_CONTEXT_POINTS)
        dataset_test = reg_data_generator.MnistNP(MAX_CONTEXT_POINTS, 784 - MAX_CONTEXT_POINTS, False)
    elif FLAGS.datasource == 'sin':
        dataset_train = reg_data_generator.SinusoidGenerator(True)
        dataset_test = reg_data_generator.SinusoidGenerator(False)

    num_iters_per_epoch = int(num_tasks / FLAGS.batch_size)
    # if not FLAGS.fine_tune:
    #     os.mkdir(save_path + 'val_img')
    if FLAGS.training:
        for e_idx in range(FLAGS.epochs):
            # for each batch tasks
            perm = np.arange(0, num_tasks) if FLAGS.datasource=='sin' else np.random.permutation(num_tasks)
            for b_idx in range(num_iters_per_epoch):
                # count iter
                index = perm[FLAGS.batch_size * b_idx:FLAGS.batch_size * (b_idx + 1)]
                if FLAGS.datasource == 'mnist':
                    train_x, train_y, valid_x, valid_y = dataset_train.generate_batch(indx=index)
                else:
                    train_x, train_y, valid_x, valid_y = dataset_train.generate_batch(indx=index)

                feed_dict = {support_x: train_x, support_y: train_y, query_x: valid_x, query_y: valid_y}

                outs = sess.run([loss, train_step, mergerd_sum, mse, Batch], feed_dict)
                train_writer.add_summary(outs[2], outs[4])
                itr = outs[4] + 1
                if itr % 10000 == 0:
                    saver.save(sess, save_path=save_path + 'model.ckpt', global_step=Batch)
    else:
        test_mse_list = []
        perm_test = np.random.permutation(num_tasks_test)
        for iii in range(int(num_tasks_test / FLAGS.batch_size)):
            index = perm_test[FLAGS.batch_size * iii:FLAGS.batch_size * (iii + 1)]
            if FLAGS.datasource == 'mnist':
                train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(indx=index)
            else:
                train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(indx=index)

            feed_dict = {support_x: train_x, support_y: train_y, query_x: valid_x, query_y: valid_y}
            outs = sess.run([mse], feed_dict)
            test_mse_list.append(outs[0])
        testing_mse = np.mean(test_mse_list)

        print('testing mse is ', testing_mse)


if __name__ == '__main__':
    if FLAGS.anp:
        ANP_training()
    else:
        NP_training()