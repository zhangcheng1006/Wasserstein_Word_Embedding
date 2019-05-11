import os
import sys
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
import networkx as nx
# import matplotlib.pyplot as plt

def cdist(X, Y):
    """
    Parameters:
    -----------
        X, Y: 2D array, [dim, ground_dim]
    Returns:
    -----------
        distances: 2D array, [dim, dim], the cost matrix
    """
    X2 = tf.reduce_sum(tf.square(X), 1)
    Y2 = tf.reduce_sum(tf.square(Y), 1)
    X2 = tf.reshape(X2, [-1, 1])
    Y2 = tf.reshape(Y2, [1, -1])
    # return pairwise euclidead difference matrix
    distances = tf.sqrt(tf.maximum(
        X2 + Y2 - 2 * tf.matmul(X, Y, False, True), 1e-5))
    # distances = tf.maximum(
    #     X2 + Y2 - 2 * tf.matmul(X, Y, False, True), 0.0)
    assert distances.shape == [X.shape[0], Y.shape[0]]
    return distances


def compute_T(K, u, v, n_iter, tol):
    """
    Parameters:
    -----------
        K: 2D array, [M, N]
        u: 1D array, [M, ]
        v: 1D array, [N, ]
        n_iter: number of iterations for matrix balancing
        tol: tolerance for stopping matrix balancing iterations
    Returns:
    ----------
        T_opt: 2D array, [M, N], the optimal transport plan
    """
    K_tilde = 1. / u * K
    r = tf.zeros([int(u.shape[0]), 1], dtype=tf.float32)
    r_new = tf.ones([int(u.shape[0]), 1], dtype=tf.float32)

    def cond(r, r_new):
        r_enter = tf.reduce_any(tf.abs(r_new - r) > tol)
        return r_enter

    def body(r, r_new):
        r = r_new
        r_new = 1. / tf.matmul(K_tilde, v / tf.matmul(K, r, True, False))
        return [r, r_new]

    _, r = tf.while_loop(cond, body, [r, r_new], maximum_iterations=n_iter)
    c = v / tf.matmul(K, r, True, False)

    T_opt = tf.matmul(tf.diag(tf.reshape(r, (-1,))),
                      tf.matmul(K, tf.diag(tf.reshape(c, (-1,)))))

    return T_opt


def wasserstein_distance(embedding1, embedding2, u, v, lambd, p, n_iter, tol):
    """
    Parameters:
    -----------
        n1, n2: int, the id of nodes
        embeddings: 3D array, [n_nodes, dim, ground_dim], the embeddings of each node
        u: 1D array, [M, ]
        v: 1D array, [N, ]
        lambd: float, the regularization parameter
        p: int, the power of ground metric
        n_iter: int, max number of iterations
        tol: float, tolerance for stopping matrix balancing iterations
    Returns:
    -----------
        distance: float, Wasserstein distance
    """
    D = cdist(embedding1, embedding2)
    D_p = tf.pow(D, p)
    K = tf.exp(-D_p / lambd)
    T = compute_T(K, u, v, n_iter, tol)

    # distance = tf.trace(tf.matmul(D_p, T, False, True)) + lambd * \
    #     tf.trace(tf.matmul(T, tf.log(T) -
    #                        tf.ones(T.shape, dtype=tf.float64), False, True))
    distance = tf.trace(tf.matmul(D_p, T, False, True))
    return distance


def wasserstein_distances(embeddings1, embeddings2, u, v, lambd, p, n_iter, tol):
    """
    Returns:
    -----------
        results: all Wasserstein distances of node pairs
    """
    results = tf.map_fn(lambda x: wasserstein_distance(x[0], x[1], u, v, lambd, p, n_iter, tol), (embeddings1, embeddings2), dtype=tf.float32)
    return results

# def wasserstein_R1_distance(n1, n2, embeddings):
#     v1 = embeddings[n1, :]
#     v2 = embeddings[n2, :]
#     distance = tf.reduce_sum(tf.abs(tf.sort(v1) - tf.sort(v2)))
#     return distance

# def wasserstein_R1_distances(pairs, embeddings):
#     results = tf.map_fn(lambda x: wasserstein_R1_distance(x[0], x[1], embeddings), pairs, dtype=tf.float64)
#     return results

# def euclidean_distance(n1, n2, embeddings):
#     """
#     Returns:
#     ------------
#         distance: Euclidean distance between two nodes
#     """
#     v1 = embeddings[n1, :]
#     v2 = embeddings[n2, :]
#     distance = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2)))
#     return distance

# def euclidean_distances(pairs, embeddings):
#     """
#     Returns:
#     -----------
#         results: all Euclidean distances of node pairs
#     """
#     results = tf.map_fn(lambda x: euclidean_distance(x[0], x[1], embeddings), pairs, dtype=tf.float64)
#     return results

# def hyperbolic_distance(n1, n2, embeddings, eps):
#     """
#     Returns:
#     ------------
#         distance: hyperbolic distance between two nodes
#     """
#     v1 = embeddings[n1, :]
#     v2 = embeddings[n2, :]
#     norm1 = tf.norm(v1)
#     norm2 = tf.norm(v2)
#     v1 = tf.cond(tf.greater_equal(norm1, 1), lambda: v1 / norm1 - eps, lambda: v1)
#     v2 = tf.cond(tf.greater_equal(norm2, 1), lambda: v2 / norm2 - eps, lambda: v2)
    
#     distance = tf.acosh(1 + 2 * tf.reduce_sum(tf.square(v1 - v2)) / ((1 - tf.reduce_sum(tf.square(v1))) * (1 - tf.reduce_sum(tf.square(v2)))))
#     return distance

# def hyperbolic_distances(pairs, embeddings, eps):
#     """
#     Returns:
#     -----------
#         results: all hyperbolic distances of node pairs
#     """
#     results = tf.map_fn(lambda x: hyperbolic_distance(x[0], x[1], embeddings, eps), pairs, dtype=tf.float64)
#     return results

def kl_distance(n1, n2, embeddings, eps):
    """
    Returns:
    ------------
        distance: KL distance between two nodes
    """
    v1 = embeddings[n1, :]
    v2 = embeddings[n2, :]
    min1 = tf.reduce_min(v1)
    min2 = tf.reduce_min(v2)
    v1 = tf.cond(tf.less_equal(min1, 0), lambda: v1 - min1 + eps, lambda: v1)
    v2 = tf.cond(tf.less_equal(min2, 0), lambda: v2 - min2 + eps, lambda: v2)
    v1 = v1 / tf.norm(v1)
    v2 = v2 / tf.norm(v2)
    kl = (tf.reduce_sum(v1 * (tf.log(v1) - tf.log(v2))) + tf.reduce_sum(v2 * (tf.log(v2) - tf.log(v1)))) / 2
    return kl

def kl_distances(pairs, embeddings, eps):
    """
    Returns:
    -----------
        results: all KL distances of node pairs
    """
    results = tf.map_fn(lambda x: kl_distance(x[0], x[1], embeddings, eps), pairs, dtype=tf.float32)
    return results

def objective(obj_distances, embed_distances, m):
    mask = obj_distances > 0
    pos = tf.square(tf.boolean_mask(embed_distances, mask))
    neg = tf.square(tf.boolean_mask(tf.maximum(m-embed_distances, 0), tf.logical_not(mask)))
    loss = tf.reduce_sum(tf.concat([pos, neg], 0))
    return loss

def negative_sampling(word_pairs, vocab2prob, rate):
    word_neg_pairs = []
    for pair in word_pairs:
        w1 = pair[0]
        neg_samples = np.random.choice(len(vocab2prob.keys()), size=rate, p=list(vocab2prob.values()))
        for sample in neg_samples:
            while sample == w1:
                sample = np.random.choice(len(vocab2prob.keys()), size=1, p=list(vocab2prob.values()))
            word_neg_pairs.append([w1, sample])
    return np.array(word_neg_pairs)
