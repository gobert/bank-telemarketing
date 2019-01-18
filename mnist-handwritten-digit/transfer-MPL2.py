import tensorflow as tf
import numpy as np

""" Hyperparameters"""
n_outputs = 5
learning_rate = 0.0001
n_epochs = 100
batch_size = 20

""" Helpers """
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


he_init = tf.variance_scaling_initializer()


def __shuffle_batch__(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

""" Load train, test & valid data """
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

X_train = X_train[y_train >= 5]
y_train = y_train[y_train >= 5] - 5
X_valid = X_valid[y_valid >= 5]
y_valid = y_valid[y_valid >= 5] - 5
X_test = X_test[y_test >= 5]
y_test = y_test[y_test >= 5] -5


def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)


X_train, y_train = sample_n_instances_per_class(X_train, y_train, n=100)
X_valid, y_valid = sample_n_instances_per_class(X_valid, y_valid, n=30)

""" build the graph """
reset_graph()

saver = tf.train.import_meta_graph('./tf_models/transfer-MLP.ckpt.meta')

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
new_hidden4 = tf.get_default_graph().get_tensor_by_name("dnn/hidden4/Elu:0")

new_hidden5 = tf.layers.dense(new_hidden4, 100, name='new_hidden5', activation=tf.nn.elu, kernel_initializer=he_init)

new_logits = tf.layers.dense(new_hidden5, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    # saver.restore(sess, './tf_models/transfer-MLP.ckpt')

    for epoch in range(n_epochs):
        for X_batch, y_batch in __shuffle_batch__(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    # save_path = new_saver.save(sess, "./my_new_model_final.ckpt")




from IPython import embed; embed()
