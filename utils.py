""" This modules implements some tool functions."""

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import tensorflow as tf

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
        X2 + Y2 - 2 * tf.matmul(X, Y, False, True), 1e-3))

    assert distances.shape == [X.shape[0], Y.shape[0]]
    return distances


def compute_T(K, u, v, n_iter, tol):
    """ Matrix balancing for computing Wasserstein distance.

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
    """ Computes Wasserstein distance between two vectors.
    """
    D = cdist(embedding1, embedding2)
    D_p = tf.pow(D, p)
    K = tf.exp(-D_p / lambd)
    T = compute_T(K, u, v, n_iter, tol)

    distance = tf.trace(tf.matmul(D_p, T, False, True))
    return distance


def wasserstein_distances(embeddings1, embeddings2, u, v, lambd, p, n_iter, tol):
    """ Computes Wasserstein distances between vector pairs.
    """
    results = tf.map_fn(lambda x: wasserstein_distance(x[0], x[1], u, v, lambd, p, n_iter, tol), (embeddings1, embeddings2), dtype=tf.float32)
    return results

def kl_distance(embedding1, embedding2, eps):
    """ Computes KL divergence between two vectors.
    """
    min1 = tf.reduce_min(embedding1)
    min2 = tf.reduce_min(embedding2)
    v1 = tf.cond(tf.less_equal(min1, 0), lambda: embedding1 - min1 + eps, lambda: embedding1)
    v2 = tf.cond(tf.less_equal(min2, 0), lambda: embedding2 - min2 + eps, lambda: embedding2)
    v1 = v1 / tf.norm(v1)
    v2 = v2 / tf.norm(v2)
    kl = (tf.reduce_sum(v1 * (tf.log(v1) - tf.log(v2))) + tf.reduce_sum(v2 * (tf.log(v2) - tf.log(v1)))) / 2
    return kl

def kl_distances(embeddings1, embeddings2, eps):
    """ Computes KL divergence between vector pairs.
    """
    results = tf.map_fn(lambda x: kl_distance(x[0], x[1], eps), (embeddings1, embeddings2), dtype=tf.float32)
    return results

def objective(obj_distances, embed_distances, m):
    """ The objective function to optimize. 
    """
    mask = obj_distances > 0
    pos = tf.square(tf.boolean_mask(embed_distances, mask))
    neg = tf.square(tf.boolean_mask(tf.maximum(m-embed_distances, 0), tf.logical_not(mask)))
    loss = tf.reduce_sum(tf.concat([pos, neg], 0))
    return loss
