import re

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from dnn_pytorch import LongTensor, FloatTensor
from dnn_pytorch.dnn_utils import init_param, log_sum_exp, sequence_mask
from loader import load_embedding
from layers_meta import meta_linear_ops


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def get_mask(batch_len, max_len=None):
    if not max_len:
        max_len = torch.max(batch_len)

    mask = Variable(torch.zeros((len(batch_len), max_len.item())))
    if batch_len.is_cuda:
        mask = mask.cuda()
    for i in range(len(batch_len)):
        mask[i, :batch_len.data[i]] = 1

    return mask


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.emb = nn.Embedding(vocab_size, dim)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        input_emb = self.emb(input)
        return input_emb

    def forward_meta(self, input, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
        if meta_loss is not None:

            if not stop_gradient:
                grad_weight = autograd.grad(meta_loss, self.emb.weight, create_graph=True)[0]

            else:
                grad_weight = Variable(autograd.grad(meta_loss, self.emb.weight, create_graph=True)[0].data,
                                       requires_grad=False)

            # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.emb.weight = self.emb.weight - grad_weight * meta_step_size

        return self.emb(input)


class EmbeddingProj(nn.Module):
    def __init__(self, mono_dim, common_dim):
        super(EmbeddingProj, self).__init__()
        self.mono_dim = mono_dim
        self.common_dim = common_dim
        self.encoder = nn.Parameter(torch.Tensor(mono_dim, common_dim))
        self.encoder_bias = nn.Parameter(torch.Tensor(common_dim))
        self.encoder_bn = nn.BatchNorm1d(common_dim, momentum=0.01)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        encoded = torch.sigmoid(
            torch.mm(input, self.encoder) + self.encoder_bias)
        encoded = self.encoder_bn(encoded)
        return encoded


# combine character-level cnn word representation with word representation
class CharCnnWordEmb(nn.Module):
    def __init__(self, word_dim, char_dim, char_conv, max_word_length, filter_withs):
        super(CharCnnWordEmb, self).__init__()

        # model parameters
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_conv = char_conv
        self.max_word_length = max_word_length
        self.filter_withs = filter_withs

        # cnn char layer
        self._cnn_char = nn.ModuleList([nn.Conv2d(1, char_conv, (w, char_dim)) for w in filter_withs])

        self.combined_word_dim = word_dim + char_conv*len(filter_withs)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_chars, char_len, input_word):
        input_chars_new = input_chars.view(input_chars.size(0)*input_chars.size(1), 1, input_chars.size(2), input_chars.size(3))
        conv_chars = [F.relu(conv(input_chars_new)).squeeze(3) for conv in self._cnn_char]
        conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]
        char_cnn_out = torch.cat(conv_chars, 1)
        char_cnn_out = char_cnn_out.view(input_chars.size(0), input_chars.size(1), char_cnn_out.size(1))
        combined_word_repr = torch.cat([input_word, char_cnn_out], dim=2)
        outputs = combined_word_repr
        combined_word_dim = self.combined_word_dim

        return outputs, combined_word_dim

    def forward_meta(self, input_chars, char_len, input_word, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
        if meta_loss is not None:

            if not stop_gradient:
                grad_weight = autograd.grad(meta_loss, self._cnn_char.weight, create_graph=True)[0]

            else:
                grad_weight = Variable(autograd.grad(meta_loss, self._cnn_char.weight, create_graph=True)[0].data,
                                       requires_grad=False)

            # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self._cnn_char.weight = self._cnn_char.weight - grad_weight * meta_step_size

        input_chars_new = input_chars.view(input_chars.size(0)*input_chars.size(1), 1, input_chars.size(2), input_chars.size(3))
        conv_chars = [F.relu(conv(input_chars_new)).squeeze(3) for conv in self._cnn_char]
        conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]
        char_cnn_out = torch.cat(conv_chars, 1)
        char_cnn_out = char_cnn_out.view(input_chars.size(0), input_chars.size(1), char_cnn_out.size(1))
        combined_word_repr = torch.cat([input_word, char_cnn_out], dim=2)
        outputs = combined_word_repr
        combined_word_dim = self.combined_word_dim

        return outputs, combined_word_dim


# encode a sequence of words with bi-lstm
class EncodeLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidrection=True, dropout=0):
        super(EncodeLstm, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidrection = bidrection

        if dropout:
            self.word_dropout = nn.Dropout(p=dropout)

        self.word_lstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=bidrection, batch_first=True)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_word_lstm_hidden(self, batch_size):
        if self.bidrection:
            return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                      .type(FloatTensor)),
                    autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                      .type(FloatTensor)))
        else:
            return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)
                                      .type(FloatTensor)),
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)
                                      .type(FloatTensor)))

    def forward(self, input, seq_len, batch_size, dropout=0):
        if dropout:
            input = self.word_dropout(input)

        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        seq_len, idx = seq_len.sort(descending=True)
        input = input[idx]

        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len.data.cpu().numpy(), batch_first=True)
        # word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        word_lstm_out, word_lstm_h = self.word_lstm(input)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        _, unsorted_idx = idx.sort()
        word_lstm_out = word_lstm_out[unsorted_idx]

        outputs = word_lstm_out

        return outputs

    def forward_meta(self, input, seq_len, batch_size, dropout=0, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
        if meta_loss is not None:

            if not stop_gradient:
                grad_weight = autograd.grad(meta_loss, self.word_lstm.weight, create_graph=True)[0]

            else:
                grad_weight = Variable(autograd.grad(meta_loss, self.word_lstm.weight, create_graph=True)[0].data,
                                       requires_grad=False)

            # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.word_lstm.weight = self.word_lstm.weight - grad_weight * meta_step_size

        if dropout:
            input = self.word_dropout(input)

        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        seq_len, idx = seq_len.sort(descending=True)
        input = input[idx]

        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len.data.cpu().numpy(), batch_first=True)
        # word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        word_lstm_out, word_lstm_h = self.word_lstm(input)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        _, unsorted_idx = idx.sort()
        word_lstm_out = word_lstm_out[unsorted_idx]

        outputs = word_lstm_out

        return outputs


# encode a sequence of words with bi-lstm
class EncodeLstmMeta(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidrection=True, dropout=0):
        super(EncodeLstmMeta, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidrection = bidrection

        if dropout:
            self.word_dropout = nn.Dropout(p=dropout)

        self.word_lstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=bidrection, batch_first=True)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_word_lstm_hidden(self, batch_size):
        if self.bidrection:
            return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                      .type(FloatTensor)),
                    autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                      .type(FloatTensor)))
        else:
            return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)
                                      .type(FloatTensor)),
                    autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)
                                      .type(FloatTensor)))

    def forward(self, input, seq_len, batch_size, dropout=0, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
        if dropout:
            input = self.word_dropout(input)

        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        seq_len, idx = seq_len.sort(descending=True)
        input = input[idx]

        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len.data.cpu().numpy(), batch_first=True)
        # word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        word_lstm_out, word_lstm_h = meta_lstm_ops(inputs=input,
                                                   weight=self.word_lstm.all_weights(),
                                                   model=self.word_lstm,
                                                   input_dim = self.input_dim,
                                                   hidden_dim = self.hidden_dim,
                                                   bidrection = self.bidrection,
                                                   meta_loss=meta_loss,
                                                   meta_step_size=meta_step_size,
                                                   stop_gradient=stop_gradient)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        _, unsorted_idx = idx.sort()
        word_lstm_out = word_lstm_out[unsorted_idx]

        outputs = word_lstm_out

        return outputs


class LinearProj(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_size):
        super(LinearProj, self).__init__()

        # model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size

        self.tanh_linear = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, label_size)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        tanh_out = nn.Tanh()(self.tanh_linear(input))
        linear_out = self.linear(tanh_out)
        outputs = linear_out
        return outputs

    def forward_meta(self, input, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
        if meta_loss is not None:

            if not stop_gradient:
                grad_weight_tanh_linear = autograd.grad(meta_loss, self.tanh_linear.weight, create_graph=True)[0]
                grad_weight_linear = autograd.grad(meta_loss, self.linear.weight, create_graph=True)[0]

            else:
                grad_weight_tanh_linear = Variable(autograd.grad(meta_loss, self.tanh_linear.weight,
                                                                 create_graph=True)[0].data, requires_grad=False)
                grad_weight_linear = Variable(autograd.grad(meta_loss, self.linear.weight,
                                                            create_graph=True)[0].data, requires_grad=False)

            # embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.tanh_linear.weight = self.tanh_linear.weight - grad_weight_tanh_linear * meta_step_size
            self.linear.weight = self.linear.weight - grad_weight_linear * meta_step_size

        tanh_out = nn.Tanh()(self.tanh_linear(input))
        linear_out = self.linear(tanh_out)
        outputs = linear_out
        return outputs


class LinearProjMeta(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_size):
        super(LinearProjMeta, self).__init__()

        # model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size

        self.tanh_linear = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, label_size)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        tanh_out = nn.Tanh()(self.tanh_linear(input))
        linear_out = self.linear(tanh_out)
        outputs = linear_out
        return outputs

    def forward_meta(self, input, meta_step_size=0.001, meta_loss=None, stop_gradient=False):

        tanh_out = nn.Tanh()(meta_linear_ops(inputs=input,
                                             weight=self.tanh_linear.weight,
                                             bias=self.tanh_linear.bias,
                                             meta_loss=meta_loss,
                                             meta_step_size=meta_step_size,
                                             stop_gradient=stop_gradient))
        linear_out = meta_linear_ops(inputs=tanh_out,
                                     weight=self.linear.weight,
                                     bias=self.linear.bias,
                                     meta_loss=meta_loss,
                                     meta_step_size=meta_step_size,
                                     stop_gradient=stop_gradient)
        outputs = linear_out
        return outputs


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class CnnDiscriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, word_dim, word_conv, filter_withs, output_dim):
        """Init discriminator."""
        super(CnnDiscriminator, self).__init__()

        self._cnn_word = nn.ModuleList([nn.Conv2d(1, word_conv, (w, word_dim)) for w in filter_withs])

        linear_input_dim = word_conv * len(filter_withs)
        # self.layer = nn.Sequential(
        #     nn.Linear(linear_input_dim, output_dim),
        #     nn.LogSoftmax()
        # )
        self.layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(linear_input_dim, output_dim),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
            #nn.LogSoftmax()
        )

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_word):
        """Forward the discriminator."""
        input_word = input_word.contiguous().view(input_word.size(0), 1, input_word.size(1), input_word.size(2))
        conv_words = [F.relu(conv(input_word)).squeeze(3) for conv in self._cnn_word]
        conv_words = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_words]
        word_cnn_out = torch.cat(conv_words, 1)
        #out = self.layer(word_cnn_out)
        out = self.layer(word_cnn_out).view(-1)

        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, dropout):
        super(Encoder, self).__init__()

        # model parameters
        self.input_dim = input_dim

        # dropout for word bi-lstm layer
        if dropout:
            self.word_dropout = nn.Dropout(p=dropout)

        # single layer encoder
        self.encoder = nn.Parameter(torch.Tensor(input_dim, input_dim).uniform_(0.0, 0.2))
        self.encoder_bias = nn.Parameter(torch.Tensor(input_dim).uniform_(0.0, 0.2))
        self.encoder_bn = nn.BatchNorm1d(input_dim, momentum=0.01)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        encoded = torch.sigmoid(torch.mm(input, self.encoder) + self.encoder_bias)
        outputs = self.encoder_bn(encoded)
        return outputs


class LanguageModelProb(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(LanguageModelProb, self).__init__()

        # model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # dropout for word bi-lstm layer
        if dropout:
            self.word_dropout = nn.Dropout(p=dropout)

        # word bi-lstm layer
        self.word_lstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim * 2, input_dim)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_word_lstm_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)),
                autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)))

    def forward(self, input, seq_len, batch_size, dropout):
        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True)
        word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        outputs = self.linear(word_lstm_out)

        return outputs


class LanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, dropout):
        super(LanguageModel, self).__init__()

        # model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # dropout for word bi-lstm layer
        if dropout:
            self.word_dropout = nn.Dropout(p=dropout)

        # word bi-lstm layer
        self.word_lstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, vocab_size)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_word_lstm_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                  .type(FloatTensor)),
                autograd.Variable(torch.randn(2, batch_size, self.hidden_dim)
                                  .type(FloatTensor)))

    def forward(self, input, seq_len, batch_size, dropout):
        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        input = torch.nn.utils.rnn.pack_padded_sequence(input, seq_len, batch_first=True)
        word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        logits = self.linear(word_lstm_out)

        outputs = logits

        return outputs


class DropOut(nn.Module):
    def __init__(self, dropout):
        super(DropOut, self).__init__()
        self.Dropout = nn.Dropout(p=dropout)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        output = self.Dropout(input)
        return output


class CRFLoss(nn.Module):
    def __init__(self, num_labels):
        super(CRFLoss, self).__init__()

        self.num_labels = num_labels
        self.START_TAG = num_labels - 2
        self.STOP_TAG = num_labels - 1

        self.transitions = Parameter(
            torch.FloatTensor(self.num_labels, self.num_labels)
        )
        self.transitions.data[self.START_TAG, :] = -10000
        # self.transitions.data[self.STOP_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, pred, tags, seq_len, decode=False, batch_mask=None, **kwargs):
        # get batch info
        batch_size, max_seq_len, _ = pred.size()

        # set a min value
        small = -1000
        # set the values of <s> and <e> labels for each token to small
        pred[:, :, -2] = small
        pred[:, :, -1] = small
        # create start and end of the sentence state
        start_state = Variable(torch.zeros(batch_size, 1, self.num_labels).fill_(small))
        start_state[:, :, self.num_labels - 2] = 0
        end_state = Variable(torch.zeros(batch_size, 1, self.num_labels))
        if self.transitions.is_cuda:
            start_state = start_state.cuda()
            end_state = end_state.cuda()

        padded_pred = torch.cat([start_state, pred, end_state], dim=1)

        # set end sentence state based on the sentence length
        padded_pred[range(len(seq_len)), seq_len.data + 1, :] = small
        padded_pred[
            range(len(seq_len)), seq_len.data + 1,
            [self.num_labels - 1] * len(seq_len)
        ] = 0

        # compute pred path score
        padded_pred = padded_pred.permute(1, 0, 2)
        paths_scores = Variable(
            torch.FloatTensor(max_seq_len + 1, batch_size, self.num_labels)
        )
        paths_indices = Variable(
            torch.LongTensor(max_seq_len + 1, batch_size, self.num_labels)
        )
        if self.transitions.is_cuda:
            paths_scores = paths_scores.cuda()
            paths_indices = paths_indices.cuda()

        previous = padded_pred[0]
        for i in range(1, len(padded_pred)):
            _previous = previous.unsqueeze(2)
            _padded_pred = padded_pred[i].unsqueeze(1)
            if decode:
                scores = _previous + _padded_pred + self.transitions
                out, out_indices = scores.max(dim=1)
                paths_indices[i - 1] = out_indices
                previous = out
            else:
                previous = log_sum_exp(
                    _previous + _padded_pred + self.transitions, dim=1
                )
            paths_scores[i - 1] = previous

        # return indices of best paths.
        if decode:
            sequence = []
            _, previous = paths_scores[-1].max(dim=1)
            for i in reversed(range(len(paths_scores))):
                previous = paths_indices[i].gather(
                    1, previous.unsqueeze(1)
                ).squeeze(1)
                sequence.append(previous)
            sequence = torch.cat(
                [s.unsqueeze(1) for s in sequence[::-1]], dim=1
            )

            masked_seq = []
            for i, s in enumerate(sequence):
                masked_seq.append(s[1:seq_len.data[i] + 1])

            return masked_seq

        # compute real path score if reference is provided
        if tags is not None:
            paths_scores = paths_scores.permute(1, 0, 2)
            # pred_paths_scores = log_sum_exp(
            #     paths_scores.gather(
            #         1, seq_len.view(-1, 1, 1).expand(paths_scores.size(0), 1, paths_scores.size(2))
            #     ).squeeze(1),
            #     dim=1
            # ).sum()
            batch_pred_path_scores = log_sum_exp(
                paths_scores.gather(
                    1, seq_len.view(-1, 1, 1).expand(paths_scores.size(0), 1, paths_scores.size(2))
                ).squeeze(1),
                dim=1
            )
            if batch_mask is not None:
                batch_pred_path_scores = batch_pred_path_scores * batch_mask
            pred_paths_scores = batch_pred_path_scores.sum()

            # Score from tags
            real_path_mask = get_mask(seq_len)

            real_path_score = pred.gather(2, tags.unsqueeze(2)).squeeze(2)
            # real_path_score = torch.sum(real_path_score * real_path_mask)
            batch_real_path_score = torch.sum(real_path_score * real_path_mask, dim=1)
            if batch_mask is not None:
                batch_real_path_score = batch_real_path_score * batch_mask
            real_path_score = torch.sum(batch_real_path_score)

            # Score from transitions
            start_tag = Variable(
                torch.LongTensor(batch_size, 1).fill_(self.num_labels - 2)
            )
            end_tag = Variable(
                torch.LongTensor(batch_size, 1).fill_(0)
            )
            if self.transitions.is_cuda:
                start_tag = start_tag.cuda()
                end_tag = end_tag.cuda()

            padded_tags_ids = torch.cat([start_tag, tags, end_tag], dim=1)

            # set end sentence state based on the sentence length
            padded_tags_ids[
                range(len(seq_len)), seq_len.data + 1
            ] = self.num_labels - 1

            # mask out padding in batch
            transition_score_mask = get_mask(seq_len + 1)

            real_transition_score = self.transitions[
                padded_tags_ids[:, range(max_seq_len + 1)],
                padded_tags_ids[:, range(1, max_seq_len + 2)]
            ]

            # real_path_score += torch.sum(
            #     real_transition_score * transition_score_mask
            # )
            batch_real_transition_score = torch.sum(real_transition_score * transition_score_mask, dim=1)
            if batch_mask is not None:
                batch_real_transition_score = batch_real_transition_score * batch_mask
            real_path_score += torch.sum(batch_real_transition_score)

            # compute loss
            loss = pred_paths_scores - real_path_score

            return loss


class CRFLossTorch(nn.Module):
    def __init__(self, num_labels):
        super(CRFLossTorch, self).__init__()

        self.num_labels = num_labels
        self.tagset_size = num_labels
        self.START_TAG = num_labels-2
        self.STOP_TAG = num_labels-1
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[self.STOP_TAG, :] = -10000

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        if torch.cuda.is_available():
            forward_var = forward_var.cuda()

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                print("forward_var", forward_var)
                print("forward_var.size", forward_var.size())
                print("trans_score", trans_score)
                print("trans_score size", trans_score.size())
                print("emit_score", emit_score)
                print("emit_score size", emit_score.size())
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var, dim=1).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        print("terminal_var", terminal_var)
        print("terminal_var size", terminal_var.size())
        alpha = log_sum_exp(terminal_var, dim=1).view(1)
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_TAG  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def forward(self, pred_logits, ref, return_best_sequence=False):
        loss = 0
        if return_best_sequence: # only use it during test
            score, tag_seq = self._viterbi_decode(pred_logits)
            return score, tag_seq
        else:
            forward_score = self._forward_alg(pred_logits)
            gold_score = self._score_sentence(pred_logits, ref)
            loss = forward_score - gold_score
        return loss

    # def forward(self, pred, ref, seq_len,
    #             viterbi=False, return_best_sequence=False, batch_mask=None):
    #     # get batch info
    #     batch_size = pred.size(0)
    #     max_seq_len = pred.size(1)
    #     label_size = pred.size(2)
    #
    #     # add padding to observations.
    #     small = -1000
    #     b_s_array = np.array(
    #         [[[small] * self.num_labels + [0, small]]] * batch_size
    #     ).astype(np.float32)
    #     b_s = Variable(torch.from_numpy(b_s_array).type(FloatTensor))
    #     right_padding_array = np.array(
    #         [[[0] * self.num_labels + [small, small]]] * batch_size
    #     ).astype(np.float32)
    #     right_padding = Variable(
    #         torch.from_numpy(right_padding_array).type(FloatTensor)
    #     )
    #     observations = torch.cat(
    #         [pred,
    #          Variable(
    #              small * torch.ones((batch_size, max_seq_len, 2)).type(FloatTensor)
    #          )],
    #         dim=2
    #     )
    #     observations = torch.cat(
    #         [b_s, observations, right_padding],
    #         dim=1
    #     )
    #
    #     # because of various length in batch, add e_s to the real end of each
    #     # sequence
    #     e_s = np.array([small] * self.num_labels + [0, 1000]).astype(np.float32)
    #     e_s_mask = np.zeros(observations.size())
    #     for i in range(batch_size):
    #         e_s_mask[i][seq_len[i] + 1] = e_s
    #     observations += Variable(torch.from_numpy(e_s_mask).type(FloatTensor))
    #
    #     # compute all path scores
    #     paths_scores = Variable(
    #         FloatTensor(max_seq_len + 1, batch_size, label_size + 2)
    #     )
    #     paths_indices = Variable(
    #         LongTensor(max_seq_len + 1, batch_size, label_size + 2)
    #     )
    #     previous = observations[:, 0]
    #     for i in range(1, observations.size(1)):
    #         obs = observations[:, i]
    #         _previous = torch.unsqueeze(previous, 2)
    #         _obs = torch.unsqueeze(obs, 1)
    #         if viterbi:
    #             scores = _previous + _obs + self.transitions
    #             out, out_indices = scores.max(dim=1)
    #             if return_best_sequence:
    #                 paths_indices[i - 1] = out_indices
    #             paths_scores[i - 1] = out
    #             previous = out
    #         else:
    #             previous = log_sum_exp(_previous + _obs + self.transitions,
    #                                    dim=1)
    #             paths_scores[i - 1] = previous
    #
    #     paths_scores = paths_scores.permute(1, 0, 2)
    #     paths_indices = paths_indices.permute(1, 0, 2)
    #
    #     batch_pred_path_scores = log_sum_exp(
    #         paths_scores[
    #             torch.from_numpy(np.arange(batch_size)).type(LongTensor),
    #             torch.from_numpy(seq_len).type(LongTensor)
    #         ],
    #         dim=1
    #     )
    #
    #     if batch_mask is not None:
    #         batch_pred_path_scores = batch_pred_path_scores * batch_mask
    #
    #     all_paths_scores = batch_pred_path_scores.sum()
    #
    #     # return indices of best paths.
    #     if return_best_sequence:
    #         sequence = []
    #         for i in range(len(paths_indices)):
    #             p_indices = paths_indices[i][:seq_len[i] + 1]
    #             p_score = paths_scores[i][:seq_len[i] + 1]
    #             _, previous = p_score[-1].max(dim=0)
    #             seq = []
    #             for j in reversed(range(len(p_score))):
    #                 s = p_indices[j]
    #                 previous = s[previous]
    #                 seq.append(previous)
    #
    #             seq = torch.cat(seq[::-1] + [p_score[-1].max(dim=0)[1]])
    #
    #             sequence.append(seq[1:-1])
    #
    #         return sequence
    #
    #     # compute real path score if reference is provided
    #     if ref is not None:
    #         # Score from tags
    #         real_path_mask = Variable(
    #             torch.from_numpy(sequence_mask(seq_len))
    #         ).type(FloatTensor)
    #         real_path_score = pred[
    #             torch.from_numpy(
    #                 np.array([np.arange(batch_size)] * max_seq_len).transpose()
    #             ).type(LongTensor),
    #             torch.from_numpy(
    #                 np.array([np.arange(max_seq_len)] * batch_size)
    #             ).type(LongTensor),
    #             ref.data
    #         ]
    #
    #         batch_real_path_score = torch.sum(real_path_score * real_path_mask, 1)
    #         if batch_mask is not None:
    #             batch_real_path_score = batch_real_path_score * batch_mask
    #
    #         real_path_score = torch.sum(batch_real_path_score)
    #
    #         # Score from transitions
    #         b_id = Variable(
    #             torch.from_numpy(
    #                 np.array([[self.num_labels]] * batch_size)
    #             ).type(LongTensor)
    #         )
    #         right_padding = Variable(torch.zeros(b_id.size())).type(LongTensor)
    #
    #         padded_tags_ids = torch.cat([b_id, ref, right_padding], dim=1)
    #
    #         # because of various length in batch, add e_id to the real end of
    #         # each sequence
    #         e_id = np.array([self.num_labels + 1])
    #         e_id_mask = np.zeros(padded_tags_ids.size())
    #         for i in range(batch_size):
    #             e_id_mask[i][seq_len[i] + 1] = e_id
    #
    #         padded_tags_ids += Variable(
    #             torch.from_numpy(e_id_mask)
    #         ).type(LongTensor)
    #
    #         # mask out padding in batch
    #         transition_score_mask = Variable(
    #             torch.from_numpy(sequence_mask(seq_len + 1))
    #         ).type(FloatTensor)
    #         real_transition_score = self.transitions[
    #             padded_tags_ids[
    #             :, torch.from_numpy(np.arange(max_seq_len + 1)).type(LongTensor)
    #             ].data,
    #             padded_tags_ids[
    #             :, torch.from_numpy(np.arange(max_seq_len + 1) + 1).type(LongTensor)
    #             ].data
    #         ]
    #
    #         batch_real_transition_score = torch.sum(real_transition_score * transition_score_mask, 1)
    #         if batch_mask is not None:
    #             batch_real_transition_score = batch_real_transition_score * batch_mask
    #
    #         real_path_score += torch.sum(batch_real_transition_score)
    #
    #         # compute loss
    #         loss = all_paths_scores - real_path_score
    #
    #         return loss


class CRFLossOld(nn.Module):
    def __init__(self, num_labels):
        super(CRFLossOld, self).__init__()

        self.num_labels = num_labels

        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.transitions.data[self.num_labels-1, :] = -10000
        self.transitions.data[self.num_labels-2, :] = -10000

    def forward(self, pred, ref, seq_len,
                viterbi=False, return_best_sequence=False, batch_mask=None):
        # get batch info
        batch_size = pred.size(0)
        max_seq_len = pred.size(1)
        label_size = pred.size(2)

        # add padding to observations.
        small = -1000
        b_s_array = np.array(
            [[[small] * self.num_labels + [0, small]]] * batch_size
        ).astype(np.float32)
        b_s = Variable(torch.from_numpy(b_s_array).type(FloatTensor))
        right_padding_array = np.array(
            [[[0] * self.num_labels + [small, small]]] * batch_size
        ).astype(np.float32)
        right_padding = Variable(
            torch.from_numpy(right_padding_array).type(FloatTensor)
        )
        observations = torch.cat(
            [pred,
             Variable(
                 small * torch.ones((batch_size, max_seq_len, 2)).type(FloatTensor)
             )],
            dim=2
        )
        observations = torch.cat(
            [b_s, observations, right_padding],
            dim=1
        )

        # because of various length in batch, add e_s to the real end of each
        # sequence
        e_s = np.array([small] * self.num_labels + [0, 1000]).astype(np.float32)
        e_s_mask = np.zeros(observations.size())
        for i in range(batch_size):
            e_s_mask[i][seq_len[i] + 1] = e_s
        observations += Variable(torch.from_numpy(e_s_mask).type(FloatTensor))

        # compute all path scores
        paths_scores = Variable(
            FloatTensor(max_seq_len + 1, batch_size, label_size + 2)
        )
        paths_indices = Variable(
            LongTensor(max_seq_len + 1, batch_size, label_size + 2)
        )
        previous = observations[:, 0]
        for i in range(1, observations.size(1)):
            obs = observations[:, i]
            _previous = torch.unsqueeze(previous, 2)
            _obs = torch.unsqueeze(obs, 1)
            if viterbi:
                scores = _previous + _obs + self.transitions
                out, out_indices = scores.max(dim=1)
                if return_best_sequence:
                    paths_indices[i - 1] = out_indices
                paths_scores[i - 1] = out
                previous = out
            else:
                previous = log_sum_exp(_previous + _obs + self.transitions,
                                       dim=1)
                paths_scores[i - 1] = previous

        paths_scores = paths_scores.permute(1, 0, 2)
        paths_indices = paths_indices.permute(1, 0, 2)

        batch_pred_path_scores = log_sum_exp(
            paths_scores[
                torch.from_numpy(np.arange(batch_size)).type(LongTensor),
                torch.from_numpy(seq_len).type(LongTensor)
            ],
            dim=1
        )

        if batch_mask is not None:
            batch_pred_path_scores = batch_pred_path_scores * batch_mask

        all_paths_scores = batch_pred_path_scores.sum()

        # return indices of best paths.
        if return_best_sequence:
            sequence = []
            for i in range(len(paths_indices)):
                p_indices = paths_indices[i][:seq_len[i] + 1]
                p_score = paths_scores[i][:seq_len[i] + 1]
                _, previous = p_score[-1].max(dim=0)
                seq = []
                for j in reversed(range(len(p_score))):
                    s = p_indices[j]
                    previous = s[previous]
                    seq.append(previous)

                seq = torch.cat(seq[::-1] + [p_score[-1].max(dim=0)[1]])

                sequence.append(seq[1:-1])

            return sequence

        # compute real path score if reference is provided
        if ref is not None:
            # Score from tags
            real_path_mask = Variable(
                torch.from_numpy(sequence_mask(seq_len))
            ).type(FloatTensor)
            real_path_score = pred[
                torch.from_numpy(
                    np.array([np.arange(batch_size)] * max_seq_len).transpose()
                ).type(LongTensor),
                torch.from_numpy(
                    np.array([np.arange(max_seq_len)] * batch_size)
                ).type(LongTensor),
                ref.data
            ]

            batch_real_path_score = torch.sum(real_path_score * real_path_mask, 1)
            if batch_mask is not None:
                batch_real_path_score = batch_real_path_score * batch_mask

            real_path_score = torch.sum(batch_real_path_score)

            # Score from transitions
            b_id = Variable(
                torch.from_numpy(
                    np.array([[self.num_labels]] * batch_size)
                ).type(LongTensor)
            )
            right_padding = Variable(torch.zeros(b_id.size())).type(LongTensor)

            padded_tags_ids = torch.cat([b_id, ref, right_padding], dim=1)

            # because of various length in batch, add e_id to the real end of
            # each sequence
            e_id = np.array([self.num_labels + 1])
            e_id_mask = np.zeros(padded_tags_ids.size())
            for i in range(batch_size):
                e_id_mask[i][seq_len[i] + 1] = e_id

            padded_tags_ids += Variable(
                torch.from_numpy(e_id_mask)
            ).type(LongTensor)

            # mask out padding in batch
            transition_score_mask = Variable(
                torch.from_numpy(sequence_mask(seq_len + 1))
            ).type(FloatTensor)
            real_transition_score = self.transitions[
                padded_tags_ids[
                :, torch.from_numpy(np.arange(max_seq_len + 1)).type(LongTensor)
                ].data,
                padded_tags_ids[
                :, torch.from_numpy(np.arange(max_seq_len + 1) + 1).type(LongTensor)
                ].data
            ]

            batch_real_transition_score = torch.sum(real_transition_score * transition_score_mask, 1)
            if batch_mask is not None:
                batch_real_transition_score = batch_real_transition_score * batch_mask

            real_path_score += torch.sum(batch_real_transition_score)

            # compute loss
            loss = all_paths_scores - real_path_score

            return loss


class Highway(nn.Module):
    def __init__(self, size, num_layers):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = F.relu(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


# combine character-level cnn word representation with word representation
class CharCnnWordEmb04(nn.Module):
    def __init__(self, word_dim, char_dim, char_conv, max_word_length, filter_withs):
        super(CharCnnWordEmb04, self).__init__()

        # model parameters
        self.word_dim = word_dim
        self.char_dim = char_dim
        self.char_conv = char_conv
        self.max_word_length = max_word_length
        self.filter_withs = filter_withs
        kernel_sizes = []
        for f in filter_withs:
            kernel_sizes.append((f, char_dim))
        self.kernel_sizes = kernel_sizes
        kernel_shape = []
        for i in range(len(self.kernel_sizes)):
            kernel_shape.append([1, self.char_conv, self.kernel_sizes[i]])
        pool_sizes = [(self.max_word_length - 2 + 1, 1),
                      (self.max_word_length - 3 + 1, 1),
                      (self.max_word_length - 4 + 1, 1)]
        self.kernel_shape = kernel_shape
        self.pool_sizes = pool_sizes

        # cnn char layer
        self._cnn_char = MultiLeNetConv2dLayer(kernel_shape, pool_sizes)

        self.combined_word_dim = word_dim + char_conv * len(filter_withs)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)


    def forward(self, input_chars, char_len, char_index_mapping, input_word, seq_len):
        # cnn_char_emb = input_chars[:, :30] # 372*25*25
        # cnn_char_emb = cnn_char_emb.contiguous().view(cnn_char_emb.size(0), 1, cnn_char_emb.size(1),
        #                                               cnn_char_emb.size(2))
        # conv_chars = [F.relu(conv(cnn_char_emb)).squeeze(3) for conv in self._cnn_char]
        # conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]
        # print("conv_chars ", conv_chars)
        # print("conv_chars ", conv_chars.size())
        # char_cnn_out = torch.cat(conv_chars, 1)
        # print("char_cnn_out", char_cnn_out)
        # print("char_cnn_out", char_cnn_out.size())
        # # char_repr = char_cnn_out
        # # char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
        # # char_repr = char_repr[char_index_mapping]
        # # char_repr_padded_seq = nn.utils.rnn.PackedSequence(char_repr, seq_len)
        # #
        # # print("char_repr_padded_seq", char_repr_padded_seq)
        # #
        # # char_repr, _ = nn.utils.rnn.pad_packed_sequence(
        # #     char_repr_padded_seq
        # # )
        # char_repr = char_cnn_out
        # print("char_repr", char_repr)
        # print("char_repr", char_repr.size())
        # print("input_word", input_word)
        # print("input_word", input_word.size())
        # combined_word_repr = torch.cat([input_word, char_repr], dim=2)
        #
        # outputs = combined_word_repr
        #
        # combined_word_dim = self.combined_word_dim

        print("input_chars", input_chars)
        print("input_chars", input_chars.size())
        print("input_word", input_word.size())
        print("input_word", input_word)
        char_cnn_out = self._cnn_char.forward(input_chars)

        char_cnn_out = torch.cat(char_cnn_out, dim=1).view(
            len(seq_len), max(seq_len.data), -1
        )
        print("char_cnn_out", char_cnn_out)
        print("char_cnn_out", char_cnn_out.size())
        combined_word_repr = torch.cat([input_word, char_cnn_out], dim=2)

        combined_word_dim = self.combined_word_dim
        outputs = combined_word_repr

        return outputs, combined_word_dim


class MultiLeNetConv2dLayer(nn.Module):
    def __init__(self, kernel_shape, pool_sizes):
        super(MultiLeNetConv2dLayer, self).__init__()

        num_conv = len(kernel_shape)
        in_channels = [item[0] for item in kernel_shape]
        out_channels = [item[1] for item in kernel_shape]
        kernel_size = [item[2] for item in kernel_shape]

        self.conv_nets = []
        self.max_pool2d = []
        for i in range(num_conv):
            conv = nn.Conv2d(in_channels[i], int(out_channels[i]), kernel_size[i])
            self.conv_nets.append(conv)

            max_pool2d = nn.MaxPool2d(pool_sizes[i])
            self.max_pool2d.append(max_pool2d)
        self.conv_nets = nn.ModuleList(self.conv_nets)
        self.max_pool2d = nn.ModuleList(self.max_pool2d)

    def forward(self, input):
        conv_out = []
        input = torch.unsqueeze(input, 1)
        for conv in self.conv_nets:
            conv_out.append(conv(input))

        pooling_out = []
        for i, pool in enumerate(self.max_pool2d):
            # squeeze the last two dimensions
            after_pool = pool(conv_out[i]).squeeze(2).squeeze(2)

            pooling_out.append(after_pool)

        return pooling_out

# ---------------------------------------- old layers ------------------------------------ #
class SharedLstm(nn.Module):
    def __init__(self, model_param):
        super(SharedLstm, self).__init__()
        self.model_param = model_param
        lstm_input_dim = model_param['shared_lstm_input_dim']*2
        lstm_hidden_dim = model_param['shared_lstm_hidden_dim']
        self.word_lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim, 1, bidirectional=True, batch_first=True)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_word_lstm_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.model_param['shared_lstm_hidden_dim'])
                                  .type(FloatTensor)),
                autograd.Variable(torch.randn(2, batch_size, self.model_param['shared_lstm_hidden_dim'])
                                  .type(FloatTensor)))

    def forward(self, input, batch_size):
        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        word_lstm_out, word_lstm_h = self.word_lstm(input, word_lstm_init_hidden)
        outputs = word_lstm_out

        return outputs


class CharCnnLstm(nn.Module):
    def __init__(self, model_param):
        super(CharCnnLstm, self).__init__()

        # model parameters
        self.model_param = model_param
        word_dim = model_param['word_dim']
        word_lstm_dim = model_param['word_lstm_dim']
        char_dim = model_param['char_dim']
        dropout = model_param['dropout']
        char_conv = model_param['char_conv']

        # initialize word lstm input dim to 0
        word_lstm_input_dim = word_dim

        # cnn char layer
        filter_withs = [2, 3, 4]
        self._cnn_char = nn.ModuleList([nn.Conv2d(1, char_conv, (w, char_dim)) for w in filter_withs])
        word_lstm_input_dim += char_conv * len(filter_withs)

        # dropout for word bi-lstm layer
        if dropout: self.word_lstm_dropout = nn.Dropout(p=dropout)

        # word bi-lstm layer
        self.word_lstm = nn.LSTM(word_lstm_input_dim, word_lstm_dim, 1, bidirectional=True, batch_first=True)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_char_lstm_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.model_param['char_lstm_dim']).type(FloatTensor)),
                autograd.Variable(torch.randn(2, 1, self.model_param['char_lstm_dim']).type(FloatTensor)))

    def init_word_lstm_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)),
                autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_word, input_chars, char_len, seq_len, char_index_mapping, batch_size):

        cnn_char_emb = input_chars[:, :30]
        cnn_char_emb = cnn_char_emb.view(cnn_char_emb.size(0), 1, cnn_char_emb.size(1), cnn_char_emb.size(2))
        conv_chars = [F.relu(conv(cnn_char_emb)).squeeze(3) for conv in self._cnn_char]
        conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]
        char_cnn_out = torch.cat(conv_chars, 1)
        char_repr = char_cnn_out
        char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
        char_repr = char_repr[char_index_mapping]  # 456 * 75
        char_repr_padded_seq = nn.utils.rnn.PackedSequence(data=char_repr, batch_sizes=seq_len)
        char_repr, _ = nn.utils.rnn.pad_packed_sequence(
            char_repr_padded_seq
        )
        word_lstm_input = torch.cat([input_word, char_repr], dim=2)

        # dropout
        if self.model_param['dropout']: word_lstm_input = self.word_lstm_dropout(word_lstm_input)

        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        word_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(word_lstm_input, seq_len, batch_first=True)
        word_lstm_out, word_lstm_h = self.word_lstm( word_lstm_input, word_lstm_init_hidden)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        outputs = word_lstm_out

        return outputs


class SeqLabeling(nn.Module):
    def __init__(self, model_param):
        super(SeqLabeling, self).__init__()

        # model parameters
        self.model_param = model_param
        word_dim = model_param['word_dim']
        word_lstm_dim = model_param['word_lstm_dim']
        char_dim = model_param['char_dim']
        dropout = model_param['dropout']
        char_conv = model_param['char_conv']
        label_size = model_param['label_size']

        # initialize word lstm input dim to 0
        word_lstm_input_dim = word_dim

        # cnn char layer
        filter_withs = [2, 3, 4]
        self._cnn_char = nn.ModuleList([nn.Conv2d(1, char_conv, (w, char_dim)) for w in filter_withs])
        word_lstm_input_dim += char_conv * len(filter_withs)

        # dropout for word bi-lstm layer
        if dropout: self.word_lstm_dropout = nn.Dropout(p=dropout)

        # word bi-lstm layer
        self.word_lstm = nn.LSTM(word_lstm_input_dim, word_lstm_dim, 1, bidirectional=True, batch_first=True)
        tanh_layer_input_dim = 2 * word_lstm_dim
        self.tanh_linear = nn.Linear(tanh_layer_input_dim, word_lstm_dim)
        self.linear = nn.Linear(word_lstm_dim, label_size)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def init_char_lstm_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.model_param['char_lstm_dim']).type(FloatTensor)),
                autograd.Variable(torch.randn(2, 1, self.model_param['char_lstm_dim']).type(FloatTensor)))

    def init_word_lstm_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)),
                autograd.Variable(torch.randn(2, batch_size, self.model_param['word_lstm_dim'])
                                  .type(FloatTensor)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_word, input_chars, char_len, seq_len, char_index_mapping, batch_size):

        cnn_char_emb = input_chars[:, :30]
        cnn_char_emb = cnn_char_emb.view(cnn_char_emb.size(0), 1, cnn_char_emb.size(1), cnn_char_emb.size(2))
        conv_chars = [F.relu(conv(cnn_char_emb)).squeeze(3) for conv in self._cnn_char]
        conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]
        char_cnn_out = torch.cat(conv_chars, 1)
        char_repr = char_cnn_out
        char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
        char_repr = char_repr[char_index_mapping]  # 456 * 75
        char_repr_padded_seq = nn.utils.rnn.PackedSequence(data=char_repr, batch_sizes=seq_len)
        char_repr, _ = nn.utils.rnn.pad_packed_sequence(
            char_repr_padded_seq
        )
        word_lstm_input = torch.cat([input_word, char_repr], dim=2)

        # dropout
        if self.model_param['dropout']: word_lstm_input = self.word_lstm_dropout(word_lstm_input)

        word_lstm_init_hidden = self.init_word_lstm_hidden(batch_size)
        word_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(word_lstm_input, seq_len, batch_first=True)
        word_lstm_out, word_lstm_h = self.word_lstm( word_lstm_input, word_lstm_init_hidden)
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(word_lstm_out, batch_first=True)

        tanh_out = nn.Tanh()(self.tanh_linear(word_lstm_out))
        linear_out = self.linear(tanh_out)

        outputs = linear_out

        return outputs


class SeqLabelingOld(nn.Module):
    def __init__(self, model_param):
        super(SeqLabelingOld, self).__init__()

        #
        # model parameters
        #
        self.model_param = model_param
        word_vocab_size = model_param['word_vocab_size']
        char_vocab_size = model_param['char_vocab_size']
        feat_vocab_size = model_param['feat_vocab_size']
        word_dim = model_param['word_dim']
        word_lstm_dim = model_param['word_lstm_dim']
        char_dim = model_param['char_dim']
        char_lstm_dim = model_param['char_lstm_dim']
        feat_dim = model_param['feat_dim']
        crf = model_param['crf']
        dropout = model_param['dropout']
        char_conv = model_param['char_conv']
        label_size = model_param['label_size']

        # initialize word lstm input dim to 0
        word_lstm_input_dim = 0

        #
        # word embedding layer
        #
        self.word_emb = nn.Embedding(word_vocab_size, word_dim)
        word_lstm_input_dim += word_dim

        #
        # char embedding layer
        #
        if char_dim:
            self.char_dim = char_dim
            self.char_emb = nn.Embedding(char_vocab_size, char_dim)

        #
        # bi-lstm char layer
        #
        if char_lstm_dim:
            self.char_lstm_init_hidden = (
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor)),
                Parameter(torch.randn(2, 1, char_lstm_dim).type(
                    FloatTensor))
            )
            self.char_lstm_dim = char_lstm_dim
            self.char_lstm = nn.LSTM(char_dim, char_lstm_dim, 1,
                                     bidirectional=True, batch_first=True)
            word_lstm_input_dim += 2 * char_lstm_dim

        # cnn char layer
        if char_conv:
            max_length = 25
            out_channel = char_conv
            kernel_sizes = [(2, char_dim), (3, char_dim), (4, char_dim)]
            kernel_shape = []
            for i in range(len(kernel_sizes)):
                kernel_shape.append([1, out_channel, kernel_sizes[i]])
            pool_sizes = [(max_length - 2 + 1, 1),
                          (max_length - 3 + 1, 1),
                          (max_length - 4 + 1, 1)]
            self.multi_convs = MultiLeNetConv2dLayer(kernel_shape, pool_sizes)
            word_lstm_input_dim += out_channel * len(kernel_sizes)

        #
        # feat dim
        #
        if feat_vocab_size:
            self.feat_emb = [nn.Embedding(v, feat_dim) for v in feat_vocab_size]
            word_lstm_input_dim += len(self.feat_emb) * feat_dim
        else:
            self.feat_emb = []

        #
        # dropout for word bi-lstm layer
        #
        if dropout:
            self.word_lstm_dropout = nn.Dropout(p=dropout)

        #
        # word bi-lstm layer
        #
        self.word_lstm_init_hidden = (
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
            Parameter(torch.randn(2, 1, word_lstm_dim).type(FloatTensor)),
        )
        self.word_lstm_dim = word_lstm_dim
        self.word_lstm = nn.LSTM(word_lstm_input_dim, word_lstm_dim, 1,
                                 bidirectional=True, batch_first=True)

        #
        # tanh layer
        #
        tanh_layer_input_dim = 2 * word_lstm_dim
        self.tanh_linear = nn.Linear(tanh_layer_input_dim,
                                     word_lstm_dim)

        #
        # linear layer before loss
        #
        self.linear = nn.Linear(word_lstm_dim, label_size)

        #
        # loss
        #
        if crf:
            self.criterion = CRFLoss(label_size)
        else:
            self.softmax = nn.Softmax()
            self.criterion = CrossEntropyLoss()

        #
        # initialize weights of each layer
        #
        self.init_weights()

    def init_weights(self):
        init_param(self.word_emb)

        if self.model_param['char_dim']:
            init_param(self.char_emb)
        if self.model_param['char_lstm_dim']:
            init_param(self.char_lstm)
            self.char_lstm.flatten_parameters()
        if self.model_param['char_conv']:
            init_param(self.multi_convs)
        if self.feat_emb:
            for f_e in self.feat_emb:
                init_param(f_e)

        init_param(self.word_lstm)
        self.word_lstm.flatten_parameters()

        init_param(self.tanh_linear)

        init_param(self.linear)

    def load_pretrained(self, id_to_word, pre_emb, word_dim, **kwargs):
        if not pre_emb:
            return

        # Initialize with pretrained embeddings
        new_weights = self.word_emb.weight.data
        print('Loading pretrained embeddings from %s...' % pre_emb)
        pretrained = {}
        emb_invalid = 0
        for i, line in enumerate(load_embedding(pre_emb)):
            if type(line) == bytes:
                try:
                    line = str(line, 'utf-8')
                except UnicodeDecodeError:
                    continue
            line = line.rstrip().split()
            if len(line) == word_dim + 1:
                pretrained[line[0]] = np.array(
                    [float(x) for x in line[1:]]
                ).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        # Lookup table initialization
        for i in range(len(id_to_word)):
            word = id_to_word[i]
            if word in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word])
                c_found += 1
            elif word.lower() in pretrained:
                new_weights[i] = torch.from_numpy(pretrained[word.lower()])
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pretrained:
                new_weights[i] = torch.from_numpy(
                    pretrained[re.sub('\d', '0', word.lower())]
                )
                c_zeros += 1
        self.word_emb.weight = nn.Parameter(new_weights)

        print('Loaded %i pretrained embeddings.' % len(pretrained))
        print('%i / %i (%.4f%%) words have been initialized with '
              'pretrained embeddings.' % (
                  c_found + c_lower + c_zeros, len(id_to_word),
                  100. * (c_found + c_lower + c_zeros) / len(id_to_word)
              ))
        print('%i found directly, %i after lowercasing, '
              '%i after lowercasing + zero.' % (
                  c_found, c_lower, c_zeros
              ))

    def forward(self, inputs):
        seq_len = inputs['seq_len']
        char_len = inputs['char_len']
        char_index_mapping = inputs['char_index_mapping']

        seq_len = np.array(seq_len)
        char_len = np.array(char_len)
        batch_size = len(seq_len)

        word_lstm_input = []
        #
        # word embeddings
        #
        words = inputs['words']

        word_emb = self.word_emb(words.type(LongTensor))

        word_lstm_input.append(word_emb)

        #
        # char embeddings
        #
        char_repr = []
        if self.model_param['char_dim']:
            chars = inputs['chars']
            char_emb = self.char_emb(chars.type(LongTensor))

        #
        # char bi-lstm embeddings
        #
        char_lstm_dim = self.model_param['char_lstm_dim']
        if self.model_param['char_lstm_dim']:
            lstm_char_emb = char_emb[:, :char_len[0]]
            char_lstm_init_hidden = (
                self.char_lstm_init_hidden[0].expand(2, len(char_len), char_lstm_dim).contiguous(),
                self.char_lstm_init_hidden[1].expand(2, len(char_len), char_lstm_dim).contiguous(),
            )
            lstm_char_emb = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_char_emb, char_len, batch_first=True
            )
            char_lstm_out, char_lstm_h = self.char_lstm(
                lstm_char_emb, char_lstm_init_hidden
            )
            char_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                char_lstm_out, batch_first=True
            )
            char_lstm_h = char_lstm_h[0].permute(1, 0, 2).contiguous().view(len(char_len), 2*char_lstm_dim)
            char_repr.append(char_lstm_h)

        #
        # char CNN embeddings
        #
        if self.model_param['char_conv']:
            cnn_char_emb = char_emb[:, :25]
            char_cnn_out = self.multi_convs(cnn_char_emb)
            char_repr += char_cnn_out

        if char_repr:
            char_repr = torch.cat(char_repr, dim=1) # 456*75

            char_index_mapping = LongTensor([char_index_mapping[k] for k in range(len(char_len))])
            char_repr = char_repr[char_index_mapping]  # 456 * 75

            char_repr_padded_seq = nn.utils.rnn.PackedSequence(data=char_repr, batch_sizes=seq_len.tolist())
            char_repr, _ = nn.utils.rnn.pad_packed_sequence(
                char_repr_padded_seq
            )   # 20*50*75

            word_lstm_input.append(char_repr)

        #
        # feat input
        #
        if self.feat_emb:
            feat_emb = []
            for i, f_e in enumerate(self.feat_emb):
                feat = inputs['feats'][:, :, i]
                feat_emb.append(f_e(feat.type(LongTensor)))
            word_lstm_input += feat_emb

        #
        # bi-directional lstm
        #
        word_lstm_dim = self.model_param['word_lstm_dim']
        word_lstm_input = torch.cat(word_lstm_input, dim=2)  # 20*50*175

        # dropout
        if self.model_param['dropout']:
            word_lstm_input = self.word_lstm_dropout(word_lstm_input)

        word_lstm_init_hidden = (
            self.word_lstm_init_hidden[0].expand(2, batch_size, word_lstm_dim).contiguous(),
            self.word_lstm_init_hidden[1].expand(2, batch_size, word_lstm_dim).contiguous()
        )

        word_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            word_lstm_input, seq_len, batch_first=True
        )
        word_lstm_out, word_lstm_h = self.word_lstm(
            word_lstm_input, word_lstm_init_hidden
        )
        word_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            word_lstm_out, batch_first=True
        )

        #
        # tanh layer
        #
        tanh_out = nn.Tanh()(self.tanh_linear(word_lstm_out))

        #
        # fully connected layer
        #
        linear_out = self.linear(tanh_out)
        #
        # softmax or crf layer
        #
        if type(self.criterion) == CrossEntropyLoss:
            outputs = torch.stack(
                [self.softmax(linear_out[i]) for i in range(batch_size)], 0
            )
        elif type(self.criterion) == CRFLoss and not self.training:
            preds = linear_out
            outputs = self.criterion(preds, None, seq_len, viterbi=True, return_best_sequence=True)
        else:
            outputs = None

        #
        # compute batch loss
        #
        loss = 0
        if self.training:
            if type(self.criterion) == CrossEntropyLoss:
                preds = outputs
            elif type(self.criterion) == CRFLoss:
                preds = linear_out
            reference = inputs['tags']

            loss = self.criterion(preds, reference, seq_len)

            loss /= batch_size

        return outputs, loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, ref, seq_len):
        batch_size = pred.size(0)
        max_seq_len = pred.size(1)

        mask = sequence_mask(seq_len)
        mask = Variable(torch.from_numpy(mask).type(FloatTensor))

        # compute cross entropy loss
        loss = - torch.log(pred)[
            torch.from_numpy(
                np.array([np.arange(batch_size)] * max_seq_len).transpose()
            ).type(LongTensor),
            torch.from_numpy(
                np.array([np.arange(max_seq_len)] * batch_size)
            ).type(LongTensor),
            ref.data
        ]
        loss = torch.sum(loss * mask)

        return loss


# class CRFLoss(nn.Module):
#     def __init__(self, num_labels):
#         super(CRFLoss, self).__init__()
#
#         self.num_labels = num_labels
#
#         self.transitions = Parameter(
#             torch.from_numpy(init_variable((num_labels+2, num_labels+2))).type(FloatTensor)
#         )
#
#     def forward(self, pred, ref, seq_len,
#                 viterbi=False, return_best_sequence=False):
#         # get batch info
#         batch_size = pred.size(0)
#         max_seq_len = pred.size(1)
#         label_size = pred.size(2)
#
#         # add padding to observations.
#         small = -1000
#         b_s_array = np.array(
#             [[[small] * self.num_labels + [0, small]]] * batch_size
#         ).astype(np.float32)
#         b_s = Variable(torch.from_numpy(b_s_array).type(FloatTensor))
#         right_padding_array = np.array(
#             [[[0] * self.num_labels + [small, small]]] * batch_size
#         ).astype(np.float32)
#         right_padding = Variable(
#             torch.from_numpy(right_padding_array).type(FloatTensor)
#         )
#         observations = torch.cat(
#             [pred,
#              Variable(
#                  small * torch.ones((batch_size, max_seq_len, 2)).type(FloatTensor)
#              )],
#             dim=2
#         )
#         observations = torch.cat(
#             [b_s, observations, right_padding],
#             dim=1
#         )
#
#         # because of various length in batch, add e_s to the real end of each
#         # sequence
#         e_s = np.array([small] * self.num_labels + [0, 1000]).astype(np.float32)
#         e_s_mask = np.zeros(observations.size())
#         for i in range(batch_size):
#             e_s_mask[i][seq_len[i]+1] = e_s
#         observations += Variable(torch.from_numpy(e_s_mask).type(FloatTensor))
#
#         # compute all path scores
#         paths_scores = Variable(
#             FloatTensor(max_seq_len+1, batch_size, label_size+2)
#         )
#         paths_indices = Variable(
#             LongTensor(max_seq_len+1, batch_size, label_size+2)
#         )
#         previous = observations[:, 0]
#         for i in range(1, observations.size(1)):
#             obs = observations[:, i]
#             _previous = torch.unsqueeze(previous, 2)
#             _obs = torch.unsqueeze(obs, 1)
#             if viterbi:
#                 scores = _previous + _obs + self.transitions
#                 out, out_indices = scores.max(dim=1)
#                 if return_best_sequence:
#                     paths_indices[i-1] = out_indices
#                 paths_scores[i-1] = out
#                 previous = out
#             else:
#                 previous = log_sum_exp(_previous + _obs + self.transitions,
#                                        dim=1)
#                 paths_scores[i-1] = previous
#
#         paths_scores = paths_scores.permute(1, 0, 2)
#         paths_indices = paths_indices.permute(1, 0, 2)
#
#         all_paths_scores = log_sum_exp(
#             paths_scores[
#                 torch.from_numpy(np.arange(batch_size)).type(LongTensor),
#                 torch.from_numpy(seq_len).type(LongTensor)
#             ],
#             dim=1
#         ).sum()
#
#         # return indices of best paths.
#         if return_best_sequence:
#             sequence = []
#             for i in range(len(paths_indices)):
#                 p_indices = paths_indices[i][:seq_len[i]+1]
#                 p_score = paths_scores[i][:seq_len[i]+1]
#                 _, previous = p_score[-1].max(dim=0)
#                 seq = []
#                 for j in reversed(range(len(p_score))):
#                     s = p_indices[j]
#                     previous = s[previous]
#                     seq.append(previous)
#
#                 seq = torch.cat(seq[::-1]+[p_score[-1].max(dim=0)[1]])
#
#                 sequence.append(seq[1:-1])
#
#             return sequence
#
#         # compute real path score if reference is provided
#         if ref is not None:
#             # Score from tags
#             real_path_mask = Variable(
#                 torch.from_numpy(sequence_mask(seq_len))
#             ).type(FloatTensor)
#             real_path_score = pred[
#                 torch.from_numpy(
#                     np.array([np.arange(batch_size)]*max_seq_len).transpose()
#                 ).type(LongTensor),
#                 torch.from_numpy(
#                     np.array([np.arange(max_seq_len)]*batch_size)
#                 ).type(LongTensor),
#                 ref.data
#             ]
#             real_path_score = torch.sum(real_path_score * real_path_mask)
#
#             # Score from transitions
#             b_id = Variable(
#                 torch.from_numpy(
#                     np.array([[self.num_labels]] * batch_size)
#                 ).type(LongTensor)
#             )
#             right_padding = Variable(torch.zeros(b_id.size())).type(LongTensor)
#
#             padded_tags_ids = torch.cat([b_id, ref, right_padding], dim=1)
#
#             # because of various length in batch, add e_id to the real end of
#             # each sequence
#             e_id = np.array([self.num_labels+1])
#             e_id_mask = np.zeros(padded_tags_ids.size())
#             for i in range(batch_size):
#                 e_id_mask[i][seq_len[i] + 1] = e_id
#
#             padded_tags_ids += Variable(
#                 torch.from_numpy(e_id_mask)
#             ).type(LongTensor)
#
#             # mask out padding in batch
#             transition_score_mask = Variable(
#                 torch.from_numpy(sequence_mask(seq_len+1))
#             ).type(FloatTensor)
#             real_transition_score = self.transitions[
#                 padded_tags_ids[
#                 :, torch.from_numpy(np.arange(max_seq_len + 1)).type(LongTensor)
#                 ].data,
#                 padded_tags_ids[
#                 :, torch.from_numpy(np.arange(max_seq_len + 1) + 1).type(LongTensor)
#                 ].data
#             ]
#             real_path_score += torch.sum(
#                 real_transition_score * transition_score_mask
#             )
#
#             # compute loss
#             loss = all_paths_scores - real_path_score
#
#             return loss

class MultiLeNetConv2dLayer(nn.Module):
    def __init__(self, kernel_shape, pool_sizes):
        super(MultiLeNetConv2dLayer, self).__init__()

        num_conv = len(kernel_shape)
        in_channels = [item[0] for item in kernel_shape]
        out_channels = [item[1] for item in kernel_shape]
        kernel_size = [item[2] for item in kernel_shape]

        self.conv_nets = []
        self.max_pool2d = []
        for i in range(num_conv):
            conv = nn.Conv2d(in_channels[i], int(out_channels[i]), kernel_size[i])
            self.conv_nets.append(conv)

            max_pool2d = nn.MaxPool2d(pool_sizes[i])
            self.max_pool2d.append(max_pool2d)
        self.conv_nets = nn.ModuleList(self.conv_nets)
        self.max_pool2d = nn.ModuleList(self.max_pool2d)

    def forward(self, input):
        conv_out = []
        input = torch.unsqueeze(input, 1)
        for conv in self.conv_nets:
            conv_out.append(conv(input))

        pooling_out = []
        for i, pool in enumerate(self.max_pool2d):
            # squeeze the last two dimensions
            after_pool = pool(conv_out[i]).squeeze(2).squeeze(2)

            pooling_out.append(after_pool)

        return pooling_out


# class Discriminator(nn.Module):
#     def __init__(self, kernel_num, word_vec_size, filter_withs):
#         super(Discriminator, self).__init__()
#         self._cnn_word = nn.ModuleList([nn.Conv2d(1, kernel_num, (w, word_vec_size)) for w in filter_withs])
#         self.word_bn = nn.BatchNorm1d(kernel_num, momentum=0.01)
#         self.discriminator = nn.Linear(kernel_num*len(filter_withs), 2)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def forward(self, input_words):  # input_chars: batshsize * char sequence length * emb
#         input_words = input_words.view(input_words.size(0), 1, input_words.size(1), input_words.size(2))
#         conv_words = [F.relu(conv(input_words)).squeeze(3) for conv in self._cnn_char]
#         conv_words = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_words]  # [(N,Co), ...]*len(Ks)
#         conv_words = torch.cat(conv_words, 1)  # dim1: batchsize  dim2: 100
#         output = nn.LogSoftmax(self.discriminator(conv_words))
#
#         return output


# Sigmoid + Cross + Tied
class ProjLanguage(nn.Module):
    def __init__(self, input_size, layer1_size, kernel_num, char_vec_size, filter_withs):
        super(ProjLanguage, self).__init__()

        self._cnn_char = nn.ModuleList([nn.Conv2d(1, kernel_num, (w, char_vec_size)) for w in filter_withs])
        character_size = kernel_num * len(filter_withs)
        self.char_bn = nn.BatchNorm1d(character_size, momentum=0.01)

        self.encoder = nn.Parameter(torch.Tensor(input_size, layer1_size).uniform_(0.0, 0.02))
        self.encoder_context = nn.Parameter(torch.Tensor(input_size, layer1_size).uniform_(0.0, 0.02))

        self.encode_bias = nn.Parameter(torch.Tensor(layer1_size).uniform_(0.0, 0.02))
        self.decode_bias = nn.Parameter(torch.Tensor(input_size).uniform_(0.0, 0.02))
        self.decode_context_bias = nn.Parameter(torch.Tensor(input_size).uniform_(0.0, 0.02))

        self.encoder_bn = nn.BatchNorm1d(layer1_size, momentum=0.01)
        self.decoder_bn = nn.BatchNorm1d(input_size, momentum=0.01)
        self.decoder_context_bn = nn.BatchNorm1d(input_size, momentum=0.01)

    def forward(self, x, x_context=None, input_chars=None, cross_encoded=None):
        if x_context is None and input_chars is None and cross_encoded is None:
            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + self.encode_bias)
            encoded = self.encoder_bn(encoded)
            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)
            return encoded, decoded
        elif x_context is None and input_chars is None and cross_encoded is not None:
            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + self.encode_bias)
            encoded = self.encoder_bn(encoded)
            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)

            cross_decoded = torch.sigmoid(torch.mm(cross_encoded, torch.transpose(self.encoder, 0, 1)) +
                                          self.decode_bias)
            cross_decoded = self.decoder_bn(cross_decoded)
            return encoded, decoded, cross_decoded
        else:
            input_chars = input_chars.view(input_chars.size(0), 1, input_chars.size(1), input_chars.size(2))
            conv_chars = [F.relu(conv(input_chars)).squeeze(3) for conv in self._cnn_char]
            conv_chars = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_chars]  # [(N,Co), ...]*len(Ks)
            conv_chars = torch.cat(conv_chars, 1)  # dim1: batchsize  dim2: 100
            conv_chars = self.char_bn(conv_chars)

            encoded = torch.sigmoid(
                torch.mm(x, self.encoder) + torch.mm(x_context, self.encoder_context) + self.encode_bias)
            encoded = self.encoder_bn(encoded)

            decoded = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder, 0, 1)) + self.decode_bias)
            decoded = self.decoder_bn(decoded)
            decoded_context = torch.sigmoid(torch.mm(encoded, torch.transpose(self.encoder_context, 0, 1)) +
                                            self.decode_context_bias)
            decoded_context = self.decoder_context_bn(decoded_context)

            if cross_encoded is not None:
                cross_decoded = torch.sigmoid(torch.mm(cross_encoded, torch.transpose(self.encoder, 0, 1)) +
                                              self.decode_bias)
                cross_decoded = self.decoder_bn(cross_decoded)
                cross_decoded_context = torch.sigmoid(
                    torch.mm(cross_encoded, torch.transpose(self.encoder_context, 0, 1))
                    + self.decode_context_bias)
                cross_decoded_context = self.decoder_context_bn(cross_decoded_context)

                return encoded, conv_chars, decoded, decoded_context, cross_decoded, cross_decoded_context

            return encoded, conv_chars, decoded, decoded_contex