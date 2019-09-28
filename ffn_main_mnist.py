from __future__ import print_function

try:
    raw_input
except:
    raw_input = input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
import random
from ffn_model import FFN

# Commands
# Task parameters
FLAGS = flags.FLAGS
flags.DEFINE_string('datasource', 'mnist', 'mnist or sinusoid')
flags.DEFINE_integer('shot', 50,
                     'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('shot_test', 50, 'Number of test samples per class per task')
flags.DEFINE_integer('seed', 100, 'Set seed')
flags.DEFINE_float('l1', 1e-5, 'Weights Penalty')
flags.DEFINE_float('l2', 1e-4, 'Weights Penalty')
# Training parameters
flags.DEFINE_integer('epochs', 500, 'Number of metatraining iterations')
flags.DEFINE_integer('batch_size', 80, 'Batchsize for metatraining')
flags.DEFINE_float('lr', 5e-4, 'Meta learning rate')
flags.DEFINE_string('savepath', 'saved_model/', 'Path to save or load models')
flags.DEFINE_string('gpu', '0', 'id of the gpu to use in the local machine')
flags.DEFINE_float('wd', 1e-6, 'weight decay')
flags.DEFINE_bool('fine_tune', False, '--')
flags.DEFINE_bool('testing', False, '--')

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
PRINT_INTERVAL = 1000
TEST_PRINT_INTERVAL = 4000

if FLAGS.datasource == 'mnist':
    num_tasks = 60000
    num_tasks_test = 10000
else:
    num_tasks = 240000
    num_tasks_test = 10000


def main():
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    model = FFN('model', num_train_samples=FLAGS.shot, num_test_samples=FLAGS.shot_test,
                l1_penalty=FLAGS.l1, l2_penalty=FLAGS.l2)

    Batch = tf.Variable(0, dtype=tf.float32)
    learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.lr, global_step=Batch,
                                decay_steps=1e5, decay_rate=0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])
    loss = model.loss + FLAGS.wd * reg_term
    train_op = optimizer.minimize(loss, global_step=Batch)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=5)

    config_str = 'FFN_'
    config_str += FLAGS.datasource + '_' + str(FLAGS.shot) + 'shot_' + 'l1' + str(FLAGS.l1) + \
                  '_l2' + str(FLAGS.l2) + '_lr' + str(FLAGS.lr) + '_wd' + str(FLAGS.wd)
    print(config_str)
    save_path = FLAGS.savepath + config_str + '/'
    if FLAGS.testing or FLAGS.fine_tune:
        saver.restore(sess, tf.train.latest_checkpoint(save_path))
        print('\nload pretrained model\n')
    import reg_data_generator
    if FLAGS.datasource == 'mnist':
        dataset = reg_data_generator.Mnist(FLAGS.shot, FLAGS.shot_test, train=True)
        dataset_test = reg_data_generator.Mnist(FLAGS.shot, 784 - FLAGS.shot, train=False)
    elif FLAGS.datasource == 'sinusoid':
        dataset = reg_data_generator.DataGenerator('sinusoid', FLAGS.shot, FLAGS.shot_test, 0, FLAGS.batch_size, False, FLAGS.num_tasks)
        dataset_test = reg_data_generator.DataGenerator('sinusoid', FLAGS.shot, FLAGS.shot_test, 0, FLAGS.batch_size, False, 10000)

    tf.summary.scalar('loss_mse', model.plain_loss)
    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('reg', model.penalty_loss)
    tf.summary.scalar('learning rate', learning_rate)

    mergerd_sum = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(save_path + 'train_writer', sess.graph)
    num_iters_per_epoch = int(num_tasks / FLAGS.batch_size)
    if not FLAGS.testing:
        for e_idx in range(FLAGS.epochs):
            # for each batch tasks
            perm = np.random.permutation(num_tasks)
            for b_idx in range(num_iters_per_epoch):
                # count iter
                index = perm[FLAGS.batch_size * b_idx:FLAGS.batch_size * (b_idx + 1)]
                if FLAGS.datasource == 'mnist':
                    train_x, train_y, valid_x, valid_y = dataset.generate_batch(indx=index)
                elif FLAGS.datasource == 'sinusoid':
                    train_x, train_y, valid_x, valid_y = dataset.generate_sinusoid_batch(True, index)

                feed_dict = {model.train_inputs: train_x, model.train_labels: train_y, model.test_inputs: valid_x, model.test_labels: valid_y}

                outs = sess.run([model.loss, train_op, mergerd_sum, model.plain_loss, model.penalty_loss, Batch], feed_dict)
                train_writer.add_summary(outs[2], outs[5])
                itr = outs[5] + 1

    else:
        penalty_loss_list = []
        val_mse_list = []
        perm_test = np.random.permutation(num_tasks_test)
        for iii in range(int(num_tasks_test / FLAGS.batch_size)):
            index = perm_test[FLAGS.batch_size * iii:FLAGS.batch_size * (iii + 1)]
            if FLAGS.datasource == 'mnist':
                train_x, train_y, valid_x, valid_y = dataset_test.generate_batch(index)
            elif FLAGS.datasource == 'sinusoid':
                train_x, train_y, valid_x, valid_y = dataset_test.generate_sinusoid_batch(False, index)
            feed_dict = {model.train_inputs: train_x, model.train_labels: train_y, model.test_inputs: valid_x,
                         model.test_labels: valid_y}
            outs = sess.run([model.loss, mergerd_sum, model.predictions, model.plain_loss, model.penalty_loss],
                            feed_dict)
            penalty_loss_list.append(outs[4])
            val_mse_list.append(outs[3])
        penalty_loss = np.mean(penalty_loss_list)
        val_mse = np.mean(val_mse_list)
        print('testing mse is ', val_mse, 'penalty loss: ', penalty_loss)


if __name__ == '__main__':
    main()