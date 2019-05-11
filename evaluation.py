import pandas as pd
import numpy as np

def read_ground_truth(file):
    ground_truth = pd.read_csv(file)
    word_pairs = zip(ground_truth['Word 1'], ground_truth['Word 2'])
    sim = ground_truth['Human (mean)']

    return word_pairs, sim

def spearman_rank_correlation(true_sim, pred_sim):
    assert true_sim.shape[0] == pred_sim.shape[0], 'number of true similarity ({}) not equal to number of predicted similarity ({})!'.format(true_sim.shape[0], pred_sim.shape[0])
    n = true_sim.shape[0]

    true_rank_idx = np.argsort(true_sim)[::-1]
    true_rank = np.arange(true_sim.shape[0])
    true_rank[true_rank_idx] = np.arange(true_sim.shape[0])

    pred_rank_idx = np.argsort(pred_sim)
    pred_rank = np.arange(pred_sim.shape[0])
    pred_rank[pred_rank_idx] = np.arange(pred_sim.shape[0])

    r = 1 - 6 * np.sum((true_rank - pred_rank) ** 2) / (n * (n*n - 1))
    return r
