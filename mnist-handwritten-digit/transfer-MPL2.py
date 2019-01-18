import tensorflow as tf
import numpy as np

saver = tf.train.import_meta_graph('./tf_models/transfer-MLP.ckpt.meta')

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")


""" Load train, test & valid data """
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

""" Re-execute the graph """
with tf.Session() as sess:
    saver.restore(sess, "./tf-models/transfer-MLP.ckpt")
    accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
    print(accuracy.eval(feed_dict={X: X_valid, y: y_valid}))
