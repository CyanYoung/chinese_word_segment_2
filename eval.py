import pickle as pk

import torch

from sklearn.metrics import f1_score, accuracy_score

from build import tensorize

from segment import models

from util import map_item


device = torch.device('cpu')

seq_len = 50

path_sent = 'feat/sent_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sents, labels, thre):
    sents, labels = tensorize([sents, labels], device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sents))
    probs = torch.squeeze(probs, dim=-1)
    mask = labels > -1
    mask_probs, mask_labels = probs.masked_select(mask), labels.masked_select(mask)
    mask_preds = mask_probs > thre
    f1 = f1_score(mask_labels, mask_preds)
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1, accuracy_score(mask_labels, mask_preds)))


if __name__ == '__main__':
    test('rnn', sents, labels, thre=0.5)
    test('s2s', sents, labels, thre=0.5)
