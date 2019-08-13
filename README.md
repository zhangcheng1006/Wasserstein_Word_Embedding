# Wasserstein Word Embedding

This project aims to apply Wasserstein embedding on Word Embedding task. Another part of this project which applies Wasserstein embedding to graph and time series embedding can be find [here](https://github.com/svenhsia/Entropic-Wasserstein-Embedding).

This project consists the following directories and files:

- folder `./data/` stores training data, including preprocessed training corpus
    - `vocab.pkl` stores vocabulary list
    - `vocab2id.pkl` stores vocab2id dictionary
    - `pos_samples.pkl` stores positive training word pairs
    - `neg_samples.pkl` stores negative word pairs, obtained by *negative sampling*
- folder `./bencmark/` stores benchmark datasets
- folder `./results/` stores trained embeddings
- script `preprocessing.py` preprocesses training texts and write the above listed `.pkl` files to `./data/`.
- script `utils.py` implements tool functions.
- script `wass_net.py` and `kl_net.py` implements respectively the main functions for Wasserstein embedding and KL embedding
- notebook `Training.ipynb` demonstrates the training processes
- notebook `Demonstration.ipynb` demonstrates the evaluation processes

**N.B. : The data won't be included in the submission**, to run the whole program, please put `simple.wiki.small.txt` file under `./data/` folder, and run:

- `preprocessing.py`
- `Training.ipynb`
- `Demonstration.ipynb`
