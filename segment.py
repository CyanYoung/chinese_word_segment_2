import pickle as pk

import numpy as np

import torch

from represent import sent2ind

from util import map_item


device = torch.device('cpu')

seq_len = 50

path_word_ind = 'feat/word_ind.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

paths = {'rnn': 'model/rnn.pkl',
         's2s': 'model/rnn_s2s.pkl'}

models = {'rnn': torch.load(map_item('rnn', paths), map_location=device),
          's2s': torch.load(map_item('rnn', paths), map_location=device)}


def predict(text, name, thre):
    text = text.strip()
    pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
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
        print('s2s: %s' % predict(text, 's2s', thre=0.5))
