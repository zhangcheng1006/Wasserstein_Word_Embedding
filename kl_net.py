""" This module implements principle functions for KL embedding. """

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
from utils import *

def Siamese_kl(X, W, dim):
    # X shape: [num_samples, 2]
    embed1 = tf.gather(W, X[:, 0]) # shape [num_samples, embed_dim]
    embed2 = tf.gather(W, X[:, 1])
    embed_distances = kl_distances(embed1, embed2, 1e-10)

    return embed1, embed2, embed_distances

def train_kl(X_train, y_train, vocab_size, pre_trained_weights=None, n_epochs=5, batch_size=64, learning_rate=0.01, dim=32, m=1):

    num_samples = X_train.shape[0]
    
    X = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='X')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')
    if pre_trained_weights is not None:
        assert pre_trained_weights.ndim == 2
        W = tf.Variable(pre_trained_weights, dtype=tf.float32, name='W')
        logging.info("Load pre-trained embeddings.")
    else:
        W = tf.Variable(tf.random.uniform([vocab_size, dim], dtype=tf.float32), name='W')
        logging.info("Initialize embeddings by random.")
    _, _, Embed_distances = Siamese_kl(X, W, dim)
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
                # if i % 1000 == 0:
                #     logging.info("Storing weights (embeddings)")
                #     tmp_embeddings = W.eval()
                #     np.savez('./tmp/{}_{}_{}'.format('KL', dim, i), embeddings=tmp_embeddings)
                if np.isnan(loss):
                    raise RuntimeError("Loss is NaN.")

    return embeddings, embed_distances

def predict_kl(X_test, embeddings, dim=32):
    num_samples = len(X_test)
    vocab_size = len(embeddings)
    X_test = np.array(X_test)
    
    X = tf.placeholder(dtype=tf.int32, shape=[num_samples, 2], name='X')
    W = tf.placeholder(dtype=tf.float32, shape=[vocab_size, dim], name='W')
    _, _, Embed_distances = Siamese_kl(X, W, dim)

    with tf.Session() as sess:
        pred_sim = sess.run(Embed_distances, feed_dict={X: X_test, W: embeddings})
    
    return pred_sim

def kl_nn(word, k, vocab, vocab2id, embeddings, dim):
    if word not in vocab2id:
        raise ValueError("{} is not recognized".format(word))
    word_id = vocab2id[word]
    pairs = [[word_id, i] for i in range(len(vocab))]
    distances = predict_kl(pairs, embeddings, dim)
    nearest_id = np.argsort(distances)[1:k+1]
    return [vocab[i] for i in nearest_id]
