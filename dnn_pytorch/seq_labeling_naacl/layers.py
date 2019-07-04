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
        char_cnn_out = self._cnn_char.forward(input_chars)

        char_cnn_out = torch.cat(char_cnn_out, dim=1).view(
            len(seq_len), max(seq_len.data), -1
        )
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

            return encoded, conv_chars, decoded, decoded_context