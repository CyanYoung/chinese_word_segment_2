import json

from segment import predict


max_len = 7

path_test = 'data/test.json'
path_label = 'data/label.json'
with open(path_test, 'rb') as f:
    texts = json.load(f)
with open(path_label, 'rb') as f:
    labels = json.load(f)


def get_cut_ind(text):
    inds = set()
    for i in range(len(text)):
        if text[i] == ' ':
            inds.add(i - len(inds))
    return inds


def test(name, texts, labels):
    count, pred_num, label_num = [0] * 3
    for text, label in zip(texts, labels):
        pred = predict(text, name, max_len)
        pred_inds, label_inds = get_cut_ind(pred), get_cut_ind(label)
        for pred_ind in pred_inds:
            if pred_ind in label_inds:
                count = count + 1
        pred_num = pred_num + len(pred_inds)
        label_num = label_num + len(label_inds)
    prec, rec = count / pred_num, count / label_num
    f1 = 2 * prec * rec / (prec + rec)
    print('\n%s - prec: %.2f - rec: %.2f - f1: %.2f' % (name, prec, rec, f1))


if __name__ == '__main__':
    test('divide', texts, labels)
    test('neural', texts, labels)
