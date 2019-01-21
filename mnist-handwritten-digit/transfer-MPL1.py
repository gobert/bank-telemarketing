import tensorflow as tf
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from datetime import datetime
import os


class InteruptionProtector(object):
    """ Save model after eah epoch to Hard Drive and restore it """
    checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./my_deep_mnist_model"

    def __init__(self, session, saver, init):
        self.session = session
        self.saver = saver
        self.init = init

    def load_if_interupted(self):
        if os.path.isfile(InteruptionProtector.checkpoint_epoch_path):
            with open(InteruptionProtector.checkpoint_epoch_path, 'rb') as f:
                start_epoch = int(f.read())
            print('Training was interrupted. Continuing at epoch', start_epoch)
            self.saver.restore(self.session, InteruptionProtector.checkpoint_path)
        else:
            start_epoch = 0
            self.session.run(self.init)
        return start_epoch

    def save_against_interuption(self, epoch):
        self.saver.save(self.session, InteruptionProtector.checkpoint_path)
        with open(InteruptionProtector.checkpoint_epoch_path, "wb") as f:
            f.write(b"%d" % (epoch + 1))


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate, n_epochs, batch_size):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._session = None

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = 'tf_logs'
        self.logdir = "{}/run-{}/".format(root_logdir, now)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            n_inputs = 28 * 28
            n_outputs = 5

            X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='X')
            y = tf.placeholder(tf.int32, shape=[None], name='y')
            training = tf.placeholder_with_default(False, shape=(), name='training')

            with tf.name_scope('dnn'):
                he_init = tf.variance_scaling_initializer()
                dropout_rate = 0.5

                hidden1 = tf.layers.dense(X, 100, activation=tf.nn.relu, name='hidden1')
                hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)

                hidden2 = tf.layers.dense(hidden1_drop, 100, activation=tf.nn.relu, name='hidden2')
                hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)

                hidden3 = tf.layers.dense(hidden2_drop, 100, activation=tf.nn.relu, name='hidden3')
                hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)

                hidden4 = tf.layers.dense(hidden3_drop, 100, activation=tf.nn.relu, name='hidden4')
                hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)

                hidden5 = tf.layers.dense(hidden4_drop, 100, activation=tf.nn.relu, name='hidden5')
                hidden5_drop = tf.layers.dropout(hidden5, dropout_rate, training=training)

                logits = tf.layers.dense(hidden5_drop, n_outputs,  kernel_initializer=he_init, name='logits')

                y_proba = tf.nn.softmax(logits, name='y_proba')

            with tf.name_scope('loss'):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y, logits=logits
                )
                loss = tf.reduce_mean(xentropy, name='loss')

            with tf.name_scope('train'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                training_op = optimizer.minimize(loss)

            with tf.name_scope('eval'):
                correct = tf.nn.in_top_k(logits, y, 1)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

            # expose important variables
            self._X, self._y = X, y
            self._y_proba, self._logits, = y_proba, logits
            self._loss, self._accuracy = loss, accuracy

            self._init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._session = tf.Session(graph=self._graph)
        self._accuracy_summary = tf.summary.scalar('acc valid', self._accuracy)
        self._accuracy_summary2 = tf.summary.scalar('acc train', self._accuracy)
        self._summary_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        self.protector = InteruptionProtector(self._session, self._saver, self._init)
        with self._session.as_default() as sess:
            start_epoch = self.protector.load_if_interupted()
            self._init.run()
            for epoch in range(start_epoch, self.n_epochs):
                batch_index = 0
                batches = self.__shuffle_batch__(X_train, y_train,
                                                 self.batch_size)
                for X_batch, y_batch in batches:
                    batch_index = batch_index + 1
                    feed_dict = {self._X: X_batch, self._y: y_batch}
                    sess.run(training_op, feed_dict=feed_dict)
                acc_train = accuracy.eval(feed_dict={self._X: X_batch,
                                                     self._y: y_batch})

                acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print(epoch, 'Train accuracy: ', acc_train, 'Valid accuracy: ', acc_valid)
                summary_str = self._accuracy_summary.eval(feed_dict={X: X_valid, y: y_valid})
                summary_str2 = self._accuracy_summary2.eval(feed_dict={X: X_train, y: y_train})
                self._summary_writer.add_summary(summary_str, epoch)
                self._summary_writer.add_summary(summary_str2, epoch)
                self.protector.save_against_interuption(epoch)
            self._summary_writer.close()
            return self

    def predict(self, X):
        if not self._session:
            raise NotFittedError(
                "This %s instance is not fitted yet" % self.__class__.__name__
            )
        with self._session.as_default():
            proba = self._y_proba.eval(feed_dict={self._X: X})
            return np.argmax(proba, axis=1)

    def __shuffle_batch__(self, X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    def close_session(self):
        if self._session:
            self._session.close()


""" Load X & y """
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train = X_train[y_train < 5]
y_train = y_train[y_train < 5]
X_valid = X_valid[y_valid < 5]
y_valid = y_valid[y_valid < 5]
X_test = X_test[y_test < 5]
y_test = y_test[y_test < 5]


""" Fine-tune model """
param_distribs = {
    'n_epochs': [10, 50, 100]
}

clf = DNNClassifier(learning_rate=0.0001, n_epochs=100, batch_size=50)
# rnd_search = RandomizedSearchCV(clf, param_distribs, n_iter=4, cv=3, verbose=2)
# rnd_search.fit(X_train, y_train)
clf.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

pred = clf.predict(X_valid)
print(accuracy_score(y_valid, pred))

""" Serialize the best estimator """
clf._saver.save(clf._session, './tf_models/transfer-MLP.ckpt')
