import os
import sys
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf

from utils import *
from preprocessing import *
from net import *
from evaluation import *

vocab, vocab2id, vocab2prob, word_pairs = preprocess('./data/simple.wiki.small.txt')
logging.info("Load vocabulary and word pairs from local file.")

file_name = 'WikiSmall'

X_train = np.array([[vocab2id[w1], vocab2id[w2]] for (w1, w2) in word_pairs])
y_train = np.array([1] * len(word_pairs))

embed_dim = 32
ground_dim = 2
n_epochs = 5
batch_size = 1024
max_try = 1

# Wass R2
logging.info("Running Wasserstein R2 embedding, embed dim={}".format(embed_dim))
try_count = 0
while try_count < max_try:
    try:
        embeddings, loss_history, time_history, embed_distances = train(
            X_train, y_train, vocab_size=len(vocab), vocab2prob=vocab2prob, dim=embed_dim, 
            learning_rate=0.05, n_epochs=n_epochs, ground_dim=2, batch_size=batch_size)
        break
    except RuntimeError:
        logging.warning("Got loss NaN")
        try_count += 1
else:
    logging.warning("Fail.")

logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'WassR2', embed_dim))
np.savez('./results/{}_{}_{}_batch'.format(file_name, 'WassR2', embed_dim), 
    vocab=vocab, vocab2id=vocab2id, embeddings=embeddings, loss=loss_history, time=time_history, 
    embed_distances=embed_distances)

word_pairs_test, true_sim = read_ground_truth('./data/wordsim353.csv')
X_test = np.array([[vocab2id[w1], vocab2id[w2]] for (w1, w2) in word_pairs_test])

pred_sim = predict(X_test, embeddings, vocab_size=len(vocab))
spearman_rank = spearman_rank_correlation(true_sim, pred_sim)

# # KL
# logging.info("Running KL embedding, embed dim={}".format(embed_dim))
# try_count = 0
# while try_count < max_try:
#     try:
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='KL', embed_dim=embed_dim, 
#             learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
#         break
#     except RuntimeError:
#         logging.warning("Got loss NaN")
#         try_count += 1
# else:
#     logging.warning("Fail.")
# if normalize_distance:
#     embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
# logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'KL', embed_dim))
# np.savez('./results/{}_{}_{}_batch'.format(file_name, 'KL', embed_dim), 
#     embeddings=embeddings, loss=loss_history, time=time_history, 
#     embed_distances=embed_distances)

# # Euclidean
# logging.info("Running Euclidean embedding, embed dim={}".format(embed_dim))
# try_count = 0
# while try_count < max_try:
#     try:
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='Euc', embed_dim=embed_dim, 
#             learning_rate=0.001, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
#         break
#     except RuntimeError:
#         logging.warning("Got loss NaN")
#         try_count += 1
# else:
#     logging.warning("Fail.")
# if normalize_distance:
#     embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
# logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'Euclidean', embed_dim))
# np.savez('./results/{}_{}_{}_batch'.format(file_name, 'Euclidean', embed_dim), 
#     embeddings=embeddings, loss=loss_history, time=time_history, 
#     embed_distances=embed_distances)

# # Hyperbolic
# logging.info("Running Hyperbolic embedding, embed dim={}".format(embed_dim))
# try_count = 0
# while try_count < max_try:
#     try:
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='Hyper', embed_dim=embed_dim, 
#             learning_rate=0.00001, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
#         break
#     except RuntimeError:
#         logging.warning("Got loss NaN")
#         try_count += 1
# else:
#     logging.warning("Fail.")
# if normalize_distance:
#     embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
# logging.info("Writing {}_{}_{}_batch_adj2 to local file".format(file_name, 'Hyperbolic', embed_dim))
# np.savez('./results/{}_{}_{}_batch_adj2'.format(file_name, 'Hyperbolic', embed_dim), 
#     embeddings=embeddings, loss=loss_history, time=time_history, 
#     embed_distances=embed_distances)
    
    
    # # Wass R3
    # logging.info("Running Wasserstein R3 embedding, embed dim={}".format(embed_dim))
    # try_count = 0
    # while try_count < max_try:
    #     try:
    #         embeddings, loss_history, time_history, embed_distances, jac = train(
    #             node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #             learning_rate=0.001, n_epochs=n_epochs, ground_dim=3, nodes=num_nodes, batch_size=batch_size)
    #         break
    #     except RuntimeError:
    #         logging.warning("Got loss NaN")
    #         try_count += 1
    # else:
    #     logging.warning("Fail.")
    # if normalize_distance:
    #     embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    # logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'WassR3', embed_dim))
    # np.savez('./results/{}_{}_{}_batch'.format(file_name, 'WassR3', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)

    # # Wass R4
    # logging.info("Running Wasserstein R4 embedding, embed dim={}".format(embed_dim))
    # try_count = 0
    # while try_count < max_try:
    #     try:
    #         embeddings, loss_history, time_history, embed_distances, jac = train(
    #             node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #             learning_rate=0.001, n_epochs=n_epochs, ground_dim=4, nodes=num_nodes, batch_size=batch_size)
    #         break
    #     except RuntimeError:
    #         logging.warning("Got loss NaN")
    #         try_count += 1
    # else:
    #     logging.warning("Fail.")
    # if normalize_distance:
    #     embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    # logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'WassR4', embed_dim))
    # np.savez('./results/{}_{}_{}_batch'.format(file_name, 'WassR4', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)

