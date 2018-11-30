import pickle as pk

import numpy as np

import torch

from represent import sent2ind

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for word, ind in label_inds.items():
        ind_labels[ind] = word
    return ind_labels


seq_len = 100

path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

oov_ind = len(embed_mat) - 1

paths = {'rnn': 'model/rnn.pkl',
         'rnn_bi': 'model/rnn_bi.pkl'}

models = {'rnn': torch.load(map_item('rnn', paths), map_location='cpu'),
          'rnn_bi': torch.load(map_item('rnn_bi', paths), map_location='cpu')}


def predict(text, name, thre):
    text = text.strip()
    pad_seq = sent2ind(text, word_inds, seq_len, oov_ind, keep_oov=True)
    sent = torch.LongTensor([pad_seq])
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sent))
    probs = probs.numpy()[0]
    probs = np.squeeze(probs, axis=-1)
    preds = probs > thre
    bound = min(len(text), seq_len)
    mask_preds = preds[-bound:]
    cands = list()
    for word, pred in zip(text, mask_preds):
        cands.append(word)
        if pred:
            cands.append(' ')
    return ''.join(cands)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('rnn: %s' % predict(text, 'rnn', thre=0.5))
        print('rnn_bi: %s' % predict(text, 'rnn_bi', thre=0.5))
