import torch
import seaborn
import spacy
import numpy as np
import torch.nn as nn
import math, copy, time
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchtext import data, datasets

seaborn.set_context(context="talk")
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_mask) == 0


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # A very 'big' negative number as mask
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = [layer for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = [SublayerConnection(size, dropout) for _ in range(2)]

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = [layer for _ in range(N)]
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = [SublayerConnection(size, dropout) for _ in range(3)]

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = [nn.Linear(d_model, d_model) for _ in range(4)]
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = q.size(0)
        q, k, v = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (q, k, v))]
        x, self.attn = attention(q, k, v, mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embedding(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.model_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position_encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0., model_dim, 2) * -(math.log(10000.0) / model_dim))
        position_encoding[:, 0::2] = torch.sin(position * div)
        position_encoding[:, 1::2] = torch.cos(position * div)
        position_encoding = position_encoding.unsqueeze(0)
        self.register_buffer('pe', position_encoding)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def create_model(src_vocab, tgt_vocab, n=6, model_dim=512, ff_dim=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, model_dim)
    ff = PositionWiseFeedForward(model_dim, ff_dim, dropout)
    position = PositionalEncoding(model_dim, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(model_dim, c(attn), c(ff), dropout), n),
                           Decoder(DecoderLayer(model_dim, c(attn), c(attn), c(ff), dropout), n),
                           nn.Sequential(Embedding(model_dim, src_vocab), c(position)),
                           nn.Sequential(Embedding(model_dim, tgt_vocab), c(position)), Generator(model_dim, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Batch(object):
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum().float()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, compute_loss):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss = compute_loss(out, batch.tgt_y, batch.n_tokens)
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        if i % 100 == 1:
            elapsed = time.time() - start
            print("Epoch: {} Loss: {} Tokens per Sec: {}".format(i, loss / batch.n_tokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)


class NoamOptimizer(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        self._step = 0
        self._rate = 0
        self.warmup = warmup
        self.factor = factor
        self.optimizer = optimizer
        self.model_size = model_size

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step

        return self.factor * self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))


def get_std_opt(model):
    return NoamOptimizer(model.src_embed[0].model_dim, 2, 4000,
                         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.sum() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))


def synth_data_gen(v, batch, n_batches):
    for i in range(n_batches):
        data = torch.from_numpy(np.random.randint(1, v, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


def tokenize(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


class ComputeLoss(object):
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        norm = norm.float()
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data.item() * norm


class Iterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_index, batch):
    src, tgt = batch.src.transpose(0, 1), batch.tgt.transpose(0, 1)
    return Batch(src, tgt, pad_index)


if __name__ == "__main__":

    MIN_FREQ = 2
    MAX_LEN = 100
    BOS_TEXT = '<s>'
    EOS_TEXT = '</s>'
    BLANK_WORD = '<blank>'
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize, init_token=BOS_TEXT, eos_token=EOS_TEXT, pad_token=BLANK_WORD)
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['src']) <= MAX_LEN)
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    m = create_model(10, 10, 2)
    print(m)

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = create_model(V, V, n=2)
    model_opt = NoamOptimizer(model.src_embed[0].model_dim, 1, 400,
                              torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(synth_data_gen(V, 30, 20), model,
                  ComputeLoss(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(synth_data_gen(V, 30, 5), model,
                        ComputeLoss(model.generator, criterion, None)).item())
