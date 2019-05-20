import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
from utils import *

def Siamese_wass(X, W, dim, ground_dim, lambd, p, n_iter, tol):
    # X shape: [num_samples, 2]
    embed1 = tf.gather(W, X[:, 0]) # shape [num_samples, embed_dim * ground_dim]
    embed2 = tf.gather(W, X[:, 1])
    embed1 = tf.reshape(embed1, shape=[-1, dim, ground_dim])
    embed2 = tf.reshape(embed2, shape=[-1, dim, ground_dim])

    u = tf.ones([dim, 1], dtype=tf.float32) / dim
    v = tf.ones([dim, 1], dtype=tf.float32) / dim
    embed_distances = wasserstein_distances(embed1, embed2, u, v, lambd, p, n_iter, tol)

    return embed1, embed2, embed_distances

# def train(X_train, y_train, vocab_size, vocab2prob, n_epochs=5, batch_size=64, learning_rate=0.01, dim=32, ground_dim=2, m=1, negative_sampling_rate=1, lambd=0.1, p=1, n_iter=20, tol=1e-5):
def train_wass(X_train, y_train, vocab_size, pre_trained_weights=None, n_epochs=5, batch_size=64, learning_rate=0.01, dim=32, ground_dim=2, m=1, lambd=0.1, p=1, n_iter=20, tol=1e-5):

    num_samples = X_train.shape[0]
    
    X = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='X')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
    # X_onehot = tf.one_hot(X, depth=vocab_size) # shape [None, 2, vocab_size]
    if pre_trained_weights is not None:
        if pre_trained_weights.ndim == 2:
            assert pre_trained_weights.shape == (vocab_size, dim*ground_dim)
        elif pre_trained_weights.ndim == 3:
            assert pre_trained_weights.shape == (vocab_size, dim, ground_dim)
            pre_trained_weights = pre_trained_weights.reshape((vocab_size, dim*ground_dim))
        else:
            raise ValueError("Pre trained weights not in right shape.")
        W = tf.Variable(pre_trained_weights, dtype=tf.float32, name='W')
        logging.info("Load pre-trained embeddings.")
    else:
        W = tf.Variable(tf.random.uniform([vocab_size, dim*ground_dim], dtype=tf.float32), name='W')
        logging.info("Initialize embeddings by random.")
    _, _, Embed_distances = Siamese_wass(X, W, dim, ground_dim, lambd, p, n_iter, tol)
    Loss = objective(y, Embed_distances, m)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(Loss)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        num_batches = int(np.ceil(num_samples / batch_size))

        indices = np.arange(num_samples)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for i in range(num_batches):
                X_batch = X_train[indices[i*batch_size:(i+1)*batch_size]]
                y_batch = y_train[indices[i*batch_size:(i+1)*batch_size]]
                num_samples_batch = X_batch.shape[0]
                
                # Running the Optimizer
                _, embeddings, embed_distances, loss = sess.run([optimizer, W, Embed_distances, Loss], feed_dict={X: X_batch, y: y_batch})
                if i % 10 == 0:
                    logging.info("Epoch No.{}/{} - Batch No.{}/{} with {} samples: Loss {}".format(epoch+1, n_epochs, i+1, num_batches, num_samples_batch, loss))
                if np.isnan(loss):
                    raise RuntimeError("Loss is NaN.")
                if i % 1000 == 0:
                    logging.info("Storing weights (embeddings)")
                    tmp_embeddings = W.eval()
                    np.savez('./tmp/{}_{}_{}_{}'.format('WassR2', dim, ground_dim, i), embeddings=tmp_embeddings)

            # assert num_samples % 2 == 0
            # num_pos_samples = num_samples // 2
            # X_pos = X_train[:num_pos_samples]
            # y_pos = y_train[:num_pos_samples]
            # loss = 0
            # embed_distances = np.zeros((num_pos_samples,), dtype=float) 
            
            # logging.info("Start computing total loss and embed_distances.")

            # num_batches_loss = int(np.ceil(num_pos_samples / batch_size))
            # for i in range(num_batches_loss):
            #     X_batch = X_pos[i*batch_size:(i+1)*batch_size]
            #     y_batch = y_pos[i*batch_size:(i+1)*batch_size]
            #     num_samples_batch = X_batch.shape[0]
            #     loss_batch, embed_distances_batch = sess.run([Loss, Embed_distances], feed_dict={X: X_batch, y: y_batch})
            #     loss += num_samples_batch / num_pos_samples * loss_batch
            #     embed_distances[i*batch_size:(i+1)*batch_size] = embed_distances_batch
            #     if i % 10 == 0:
            #         logging.info("Batch No.{}/{} with {} samples: batch loss {}, cumulative loss {}".format(i+1, num_batches_loss, num_samples_batch, loss_batch, loss))
                
            # # # Displaying result on current Epoch
            # logging.info("Epoch: {}/{}, loss: {}".format(epoch+1, n_epochs, loss))

    return embeddings, embed_distances

def predict_wass(X_test, embeddings, vocab_size, dim=32, ground_dim=2, lambd=0.05, p=1, n_iter=20, tol=1e-5):
    num_samples = X_test.shape[0]
    
    X = tf.placeholder(dtype=tf.int32, shape=[num_samples, 2], name='X')
    W = tf.placeholder(dtype=tf.float32, shape=[vocab_size, dim*ground_dim], name='W')
    # X_onehot = tf.one_hot(X, depth=vocab_size)
    _, _, Embed_distances = Siamese_wass(X, W, dim, ground_dim, lambd, p, n_iter, tol)

    with tf.Session() as sess:
        pred_sim = sess.run(Embed_distances, feed_dict={X: X_test, W: embeddings})
    
    return pred_sim
