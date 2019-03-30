import torch.nn as nn


class Rnn(nn.Module):
    def __init__(self, embed_mat):
        super(Rnn, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.ra = nn.GRU(embed_len, 200, batch_first=True, bidirectional=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, 1))

    def forward(self, x):
        x = self.embed(x)
        h, h_n = self.ra(x)
        return self.dl(h)


class S2S(nn.Module):
    def __init__(self, embed_mat):
        super(S2S, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.encode = nn.GRU(embed_len, 200, batch_first=True, bidirectional=True)
        self.decode = nn.GRU(embed_len, 200, batch_first=True, bidirectional=True)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, 1))

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        h2, h2_n = self.decode(x, h1_n)
        return self.dl(h2)
