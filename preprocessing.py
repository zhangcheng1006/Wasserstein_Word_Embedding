import pickle

import numpy as np

def compute_word_prob(vocab2occ, total_occ, add_unk=False):
    # for key, value in vocab2occ.items():
    #     if key == '<unk>':
    #         print("Unknow vocabulary encountered.")
    #         vocab2occ[key] = 1.0
    #     else:
    #         prob = value / total_occ
    #         vocab2occ[key] = (np.sqrt(prob / 0.001) + 1) * 0.001 / prob
    
    subsamp_prob = lambda p: np.sqrt(0.001 / p) + 0.001 / p
    vocab2prob = {key: subsamp_prob(value / total_occ) for key, value in vocab2occ.items() if key != '<unk>'}
    if add_unk:
        vocab2prob['<unk>'] = 1.0

    return vocab2prob

def subsampling(line, vocab2prob, vocab2occ, total_occ):
    words = line.strip().split()
    remain_words = []
    for word in words:
        p = vocab2prob[word]
        eps = np.random.rand()
        if eps <= p:
            remain_words.append(word)
        else:
            vocab2occ[word] -= 1
            total_occ -= 1
    return remain_words, total_occ

def get_word_pairs(words, word_pairs, vocab2id, window_size=2):
    for i in range(len(words)-1):
        if '<unk>' not in words[i:i+2]:
            word_pairs.append((vocab2id[words[i]], vocab2id[words[i+1]]))
        if i < len(words) - 2:
            if '<unk>' not in [words[i], words[i+2]]:
                word_pairs.append((vocab2id[words[i]], vocab2id[words[i+2]]))

def preprocess(file, window_size=2):
    total_occ = 0
    unk_count = 0
    vocab = []
    idx = 0
    vocab2id = {}
    vocab2occ = {}
    with open(file, 'r') as f:
        for line_count, line in enumerate(f):
            if line_count % 10000 == 0:
                print("Treating sentence No.{}".format(line_count))
            words = line.strip().split()
            for word in words:
                total_occ += 1
                if word == '<unk>':
                    unk_count += 1
                    continue
                if vocab2id.get(word) is None:
                    vocab.append(word)
                    vocab2id[word] = idx
                    idx += 1
                    vocab2occ[word] = 1
                else:
                    vocab2occ[word] += 1

    assert np.sum(list(vocab2occ.values())) + unk_count == total_occ, 'sum of vocab occurences ({}) not equal to total occurence ({})!'.format(np.sum(list(vocab2occ.values())), total_occ)
    subsampling_prob = compute_word_prob(vocab2occ, total_occ, add_unk=True)

    # final_count = 0
    idx = 0
    # final_vocab = []
    # final_vocab2id = {}
    # final_vocab2occ = {}
    word_pairs = []
    with open(file, 'r') as f:
        for line_count, line in enumerate(f):
            if line_count % 10000 == 0:
                print("Treating sentence No.{}".format(line_count))
            # words = line.strip().split()
            remain_words, total_occ = subsampling(line, subsampling_prob, vocab2occ, total_occ)
            get_word_pairs(remain_words, word_pairs, vocab2id, window_size)

    assert np.sum(list(vocab2occ.values())) + unk_count == total_occ, 'sum of final vocab occurences ({}) not equal to final total occurence ({})!'.format(np.sum(list(vocab2occ.values())), total_occ)
    
    total_count = sum(vocab2occ.values())
    neg_sampling_prob = [vocab2occ[w]/total_count for w in vocab]

    print("Generating negative samples.")
    neg_samples = np.random.choice(len(vocab), size=len(word_pairs), p=neg_sampling_prob).tolist()
    neg_samples = list(zip([i for i, _ in word_pairs], neg_samples))
    
    return vocab, vocab2id, word_pairs, neg_samples

if __name__ == "__main__":
    print("running main function in preprocessin.py")
    # vocab, vocab2id, pos_samples, neg_samples = preprocess('./data/simple.wiki.small.txt')
    # print("Writing to local files.")
    # with open('./data/vocab.pkl', 'wb') as outfile:
    #     pickle.dump(vocab, outfile)
    # with open('./data/vocab2id.pkl', 'wb') as outfile:
    #     pickle.dump(vocab2id, outfile)
    # with open('./data/pos_samples.pkl', 'wb') as outfile:
    #     pickle.dump(pos_samples, outfile)
    # with open('./data/neg_samples.pkl', 'wb') as outfile:
    #     pickle.dump(neg_samples, outfile)

    vocab, vocab2id, pos_samples, neg_samples = preprocess('./data/simple.wiki.large.txt')
    print("Writing to local files.")
    with open('./data/vocab_large.pkl', 'wb') as outfile:
        pickle.dump(vocab, outfile)
    with open('./data/vocab2id_large.pkl', 'wb') as outfile:
        pickle.dump(vocab2id, outfile)
    with open('./data/pos_samples_large.pkl', 'wb') as outfile:
        pickle.dump(pos_samples, outfile)
    with open('./data/neg_samples_large.pkl', 'wb') as outfile:
        pickle.dump(neg_samples, outfile)

