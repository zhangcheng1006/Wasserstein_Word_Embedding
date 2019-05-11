import numpy as np

def compute_subsampling_prob(vocab2occ, total_occ):
    for key, value in vocab2occ.items():
        if key == '<unk>':
            print("Unknow vocabulary encountered.")
            vocab2occ[key] = 1.0
        else:
            prob = value / total_occ
            vocab2occ[key] = (np.sqrt(prob / 0.001) + 1) * 0.001 / prob
    return vocab2occ

def subsampling(words, vocab2prob):
    subwords = []
    for word in words:
        p = vocab2prob[word]
        eps = np.random.rand()
        if eps <= p:
            subwords.append(word)
    return subwords

def preprocess(file, window_size=2):
    total_occ = 0
    vocab = []
    idx = 0
    vocab2id = {}
    vocab2occ = {}
    with open(file, 'r') as f:
        for _, line in enumerate(f):
            words = line.strip("\n").split(" ")
            for i, word in enumerate(words):
                total_occ += 1
                if vocab2id.get(word) is None:
                    vocab.append(word)
                    vocab2id[word] = idx
                    idx += 1
                    vocab2occ[word] = 1
                else:
                    vocab2occ[word] += 1

    assert np.sum(list(vocab2occ.values())) == total_occ, 'sum of vocab occurences ({}) not equal to total occurence ({})!'.format(np.sum(list(vocab2occ.values())), total_occ)
    vocab2prob = compute_subsampling_prob(vocab2occ, total_occ)

    final_count = 0
    idx = 0
    final_vocab = []
    final_vocab2id = {}
    final_vocab2occ = {}
    word_pairs = []
    with open(file, 'r') as f:
        for _, line in enumerate(f):
            words = line.strip("\n").split(" ")
            words = subsampling(words, vocab2prob)
            for i, word in enumerate(words):
                if word == '<unk>':
                    continue
                final_count += 1
                if final_vocab2id.get(word) is None:
                    final_vocab.append(word)
                    final_vocab2id[word] = idx
                    idx += 1
                    final_vocab2occ[word] = 1
                else:
                    final_vocab2occ[word] += 1
                for j in range(i-window_size, i+window_size+1):
                    if j >= 0 and j < len(words) and j != i and words[j] != '<unk>':
                        word_pairs.append((word, words[j]))

    assert np.sum(list(final_vocab2occ.values())) == final_count, 'sum of final vocab occurences ({}) not equal to final total occurence ({})!'.format(np.sum(list(final_vocab2occ.values())), final_count)
    
    for (key, value) in final_vocab2occ.items():
        final_vocab2occ[key] = value / final_count
    
    return final_vocab, final_vocab2id, final_vocab2occ, word_pairs

# # test
# vocab, vocab2id, vocab2prob, word_pairs = preprocess('./data/simple.wiki.small.txt')
# print(len(vocab))
# print(vocab[11])
# print(vocab2id[vocab[11]])
# print(vocab2prob[vocab[11]])
# print(len(word_pairs))
# print(word_pairs[101])