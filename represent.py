import json
import pickle as pk

import re

import numpy as np

from gensim.corpora import Dictionary


embed_len = 200
min_freq = 3
max_vocab = 5000
seq_len = 100

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'


def convert(texts):
    sents, labels = list(), list()
    for text in texts:
        sent = re.sub(' ', '', text)
        sents.append(sent)
        inds, count = [0] * len(sent), 0
        for i in range(len(text)):
            if text[i] == ' ':
                count = count + 1
                inds[i - count] = 1
        labels.append(inds)
    return sents, labels


def embed(sent_words, path_word_ind, path_word_vec, path_embed):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, keep_n=max_vocab)
    word_inds = model.token2id
    with open(path_word_ind, 'wb') as f:
        pk.dump(word_inds, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind + 1] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def sent2ind(words, word_inds, seq_len, oov_ind, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word] + 1)
        elif keep_oov:
            seq.append(oov_ind)
    return pad(seq, seq_len, val=0)


def pad(seq, seq_len, val):
    if len(seq) < seq_len:
        return [val] * (seq_len - len(seq)) + seq
    else:
        return seq[-seq_len:]


def align(sent_words, labels, path_sent, path_label):
    with open(path_word_ind, 'rb') as f:
        word_inds = pk.load(f)
    with open(path_embed, 'rb') as f:
        embed_mat = pk.load(f)
    oov_ind = len(embed_mat) - 1
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, word_inds, seq_len, oov_ind, keep_oov=True)
        pad_seqs.append(pad_seq)
    pad_seqs = np.array(pad_seqs)
    pad_inds = list()
    for label in labels:
        pad_ind = pad(label, seq_len, val=-1)
        pad_inds.append(pad_ind)
    pad_inds = np.array(pad_inds)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(pad_inds, f)


def vectorize(path_data, path_sent, path_label, mode):
    with open(path_data, 'r') as f:
        texts = json.load(f)
    sents, labels = convert(texts)
    sent_words = [list(sent) for sent in sents]
    if mode == 'train':
        embed(sent_words, path_word_ind, path_word_vec, path_embed)
    align(sent_words, labels, path_sent, path_label)


if __name__ == '__main__':
    path_data = 'data/train.json'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_sent, path_label, 'train')
    path_data = 'data/dev.json'
    path_sent = 'feat/sent_dev.pkl'
    path_label = 'feat/label_dev.pkl'
    vectorize(path_data, path_sent, path_label, 'dev')
    path_data = 'data/test.json'
    path_sent = 'feat/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_sent, path_label, 'test')
