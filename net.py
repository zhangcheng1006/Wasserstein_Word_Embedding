import numpy as np
import tensorflow as tf
from utils import *

def Siamese(X, W, dim, ground_dim, lambd, p, n_iter, tol):
    x1 = X[:, 0, :]
    x2 = X[:, 1, :]
    embed1 = tf.matmul(x1, W)
    embed2 = tf.matmul(x2, W)
    embed1 = tf.reshape(embed1, shape=[-1, dim, ground_dim])
    embed2 = tf.reshape(embed2, shape=[-1, dim, ground_dim])

    u = tf.ones([dim, 1], dtype=tf.float32) / dim
    v = tf.ones([dim, 1], dtype=tf.float32) / dim
    embed_distances = wasserstein_distances(embed1, embed2, u, v, lambd, p, n_iter, tol)

    return embed1, embed2, embed_distances

def train(X_train, y_train, vocab_size, vocab2prob, n_epochs=5, batch_size=64, learning_rate=0.01, dim=32, ground_dim=2, m=1, negative_sampling_rate=1, lambd=0.05, p=1, n_iter=20, tol=1e-5):
    num_samples = X_train.shape[0]
    
    X = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='X')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
    X_onehot = tf.one_hot(X, depth=vocab_size)
    W = tf.Variable(tf.random.uniform([vocab_size, dim*ground_dim], dtype=tf.float32), name='W')
    Embed1, Embed2, Embed_distances = Siamese(X_onehot, W, dim, ground_dim, lambd, p, n_iter, tol)
    Loss = objective(y, Embed_distances, m)
    Jac = tf.gradients(Loss, W)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # Lists for storing the changing Cost and Accuracy in every Epoch
        loss_history = []
        time_history = []

        num_batches = num_samples // batch_size

        indexes = np.arange(num_samples)

        start_time = time()
        for epoch in range(n_epochs):
            X_total = None
            y_total = None
            np.random.shuffle(indexes)

            for i in range(num_batches):
                X_batch = X_train[indexes[i*batch_size:(i+1)*batch_size]]
                y_batch = y_train[indexes[i*batch_size:(i+1)*batch_size]]
                X_neg = negative_sampling(X_batch, vocab2prob, negative_sampling_rate)
                y_neg = np.array([0] * X_neg.shape[0])
                X_batch = np.concatenate((X_batch, X_neg))
                y_batch = np.concatenate((y_batch, y_neg))

                if X_total is None:
                    X_total = X_batch
                    y_total = y_batch
                else:
                    X_total = np.concatenate((X_total, X_batch))
                    y_total = np.concatenate((y_total, y_batch))
                
                # Running the Optimizer
                _, embeddings, embed1, embed2, embed_distances, loss, jac = sess.run([optimizer, W, Embed1, Embed2, Embed_distances, Loss, Jac], feed_dict={X: X_batch, y: y_batch})
                if i % 10 == 0:
                    print("Batch {}: Loss {}".format(i+1, loss))
                if np.isnan(loss):
                    raise RuntimeError("Loss is NaN.")

            if (i+1)*batch_size < num_samples:
                X_batch = X_train[indexes[(i+1)*batch_size:]]
                y_batch = y_train[indexes[(i+1)*batch_size:]]
                X_neg = negative_sampling(X_batch, vocab2prob, negative_sampling_rate)
                y_neg = np.array([0] * X_neg.shape[0])
                X_batch = np.concatenate((X_batch, X_neg))
                y_batch = np.concatenate((y_batch, y_neg))

                X_total = np.concatenate((X_total, X_batch))
                y_total = np.concatenate((y_total, y_batch))
                # Running the Optimizer
                _, embeddings, embed_distances, loss = sess.run([optimizer, W, Embed_distances, Loss], feed_dict={X: X_batch, y: y_batch})
                if np.isnan(loss):
                    raise RuntimeError("Loss is NaN.")      

            loss, embed_distances = sess.run([Loss, Embed_distances], feed_dict={X: X_total, y: y_total})
            loss_history.append(loss)
            time_history.append(time() - start_time)
            # # Displaying result on current Epoch
            logging.info("Epoch: {}/{}, loss: {}".format(epoch+1, n_epochs, loss))
            # Early stopping check
            if epoch > 2 and np.mean(loss_history[-3:-1]) - loss_history[-1] < 1e-4:
                logging.info("Early Stopped: 5 consecutive epochs with loss improvement {}".format(loss_history[-2]-loss_history[-1]))
                break
    return embeddings, loss_history, time_history, embed_distances

def predict(X, W):
    embed = tf.matmul(X, W)
    
    return embed
