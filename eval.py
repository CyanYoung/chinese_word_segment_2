import json
import pickle as pk

import re

import numpy as np

import torch

from sklearn_crfsuite.metrics import flat_f1_score, flat_accuracy_score

from util import map_item


seq_len = 100

path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
path_text = 'data/test.json'
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_text, 'rb') as f:
    texts = json.load(f)

paths = {'rnn': 'model/rnn.pkl',
         'rnn_bi': 'model/rnn_bi.pkl'}

models = {'rnn': torch.load(map_item('rnn', paths), map_location='cpu'),
          'rnn_bi': torch.load(map_item('rnn_bi', paths), map_location='cpu')}


def test(name, sents, labels, texts, thre):
    sents = torch.LongTensor(sents)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sents))
    probs = probs.numpy()
    probs = np.squeeze(probs, axis=-1)
    preds = probs > thre
    mask_labels, mask_preds = list(), list()
    for text, pred, label in zip(texts, preds, labels):
        bound = min(len(re.sub(' ', '', text)), seq_len)
        mask_preds.append(pred[-bound:])
        mask_labels.append(label[-bound:])
    print('\n%s f1: %.2f - acc: %.2f' % (name, flat_f1_score(mask_labels, mask_preds),
                                         flat_accuracy_score(mask_labels, mask_preds)))


if __name__ == '__main__':
    test('rnn', sents, labels, texts, thre=0.5)
    test('rnn_bi', sents, labels, texts, thre=0.5)
