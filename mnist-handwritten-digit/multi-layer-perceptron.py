import tensorflow as tf
import numpy as np
from datetime import datetime
import os

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}/".format(root_logdir, now)


n_inputs = 28 * 28
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
y = tf.placeholder(tf.int32, shape=[None], name='y')


# TODO understand how we reduce maximize
def neuron_layer(X, n_neurons, name='layer', activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        z = tf.matmul(X, W) + b
        if activation:
            return activation(z)
        else:
            return z


with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='n_hidden1', activation=tf.nn.relu)
    logits = neuron_layer(hidden1, n_outputs, name='outputs')


with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01  # HYPERPARAMETER
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # TODO inspect with embed


accuracy_summary = tf.summary.scalar('accuracy', accuracy)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
saver = tf.train.Saver()


"""
    Execution
"""
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 10
batch_size = 50


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


class InteruptionProtector(object):
    """ Save model after eah epoch to Hard Drive and restore it """
    checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./my_deep_mnist_model"

    def __init__(self, session):
        self.session = session

    def load_if_interupted(self):
        if os.path.isfile(InteruptionProtector.checkpoint_epoch_path):
            with open(InteruptionProtector.checkpoint_epoch_path, 'rb') as f:
                start_epoch = int(f.read())
            print('Training was interrupted. Continuing at epoch', start_epoch)
            saver.restore(self.session, InteruptionProtector.checkpoint_path)
        else:
            start_epoch = 0
            self.session.run(init)
        return start_epoch

    def save_against_interuption(self):
        with open(InteruptionProtector.checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))


tf_sess = tf.Session()
protector = InteruptionProtector(tf_sess)
with tf_sess.as_default():
    start_epoch = protector.load_if_interupted()
    init.run()

    for epoch in range(start_epoch, n_epochs):
        batch_index = 0
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            if batch_index % 10 == 0:
                step = epoch * batch_size + batch_index
                summary_str = accuracy_summary.eval(feed_dict={X: X_batch, y: y_batch})
                summary_writer.add_summary(summary_str, step)
            batch_index = batch_index + 1
            tf_sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, 'Train accuracy: ', acc_train, 'Validation Accuracy', acc_val)
        protector.save_against_interuption()
        summary_writer.close()
