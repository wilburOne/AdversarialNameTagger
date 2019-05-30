import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from nlinalg import log_sum_exp
from misc import get_mask


class CRFLayer(nn.Module):
    def __init__(self, input_size, num_labels):
        super(CRFLayer, self).__init__()

        self.num_labels = num_labels

        self.linear = nn.Linear(input_size, self.num_labels)

        self.transitions = Parameter(
            torch.FloatTensor(self.num_labels, self.num_labels)
        )

    def forward(self, input, tags, seq_len, decode=False, **kwargs):
        # get batch info
        batch_size, max_seq_len, _ = input.size()

        # shape = [batch_size, seq_len, num_labels]
        pred = self.linear(input)

        # set a min value
        small = -1000
        # set the values of <s> and <e> labels for each token to small
        pred[:, :, -2] = small
        pred[:, :, -1] = small
        # create start and end of the sentence state
        start_state = Variable(torch.zeros(batch_size, 1, self.num_labels).fill_(small))
        start_state[:, :, self.num_labels-2] = 0
        end_state = Variable(torch.zeros(batch_size, 1, self.num_labels))
        if self.transitions.is_cuda:
            start_state = start_state.cuda()
            end_state = end_state.cuda()

        padded_pred = torch.cat([start_state, pred, end_state], dim=1)

        # set end sentence state based on the sentence length
        padded_pred[range(len(seq_len)), seq_len.data+1, :] = small
        padded_pred[
            range(len(seq_len)), seq_len.data+1,
            [self.num_labels-1]*len(seq_len)
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
                paths_indices[i-1] = out_indices
                previous = out
            else:
                previous = log_sum_exp(
                    _previous + _padded_pred + self.transitions, dim=1
                )
            paths_scores[i-1] = previous

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
                masked_seq.append(s[1:seq_len.data[i]+1])

            return masked_seq

        # compute real path score if reference is provided
        if tags is not None:
            paths_scores = paths_scores.permute(1, 0, 2)
            pred_paths_scores = log_sum_exp(
                paths_scores.gather(
                    1, seq_len.view(-1, 1, 1).expand(paths_scores.size(0), 1, paths_scores.size(2))
                ).squeeze(1),
                dim=1
            ).sum()

            # Score from tags
            real_path_mask = get_mask(seq_len)

            real_path_score = pred.gather(2, tags.unsqueeze(2)).squeeze(2)
            real_path_score = torch.sum(real_path_score * real_path_mask)

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
            real_path_score += torch.sum(
                real_transition_score * transition_score_mask
            )

            # compute loss
            loss = pred_paths_scores - real_path_score

            return loss


class EnsembleCRFLayer(nn.Module):
    def __init__(self, crf_layers):
        super(EnsembleCRFLayer, self).__init__()
        self.crf_layers = crf_layers
        self.num_labels = crf_layers[0].num_labels

        input_dim = self.crf_layers[0].linear.in_features
        self.ensemble_crf = CRFLayer(input_dim, self.num_labels-2)

        self.input_linear = nn.Linear(len(crf_layers), 1)
        self.transition_linear = nn.Linear(len(crf_layers), 1)

        self.stacked_transitions = torch.stack(
            [l.transitions for l in self.crf_layers]
        )
        self.stacked_transitions = self.stacked_transitions.detach()

    def forward(self, input, tags, seq_len, **kwargs):
        self.ensemble_crf.transitions.data = self.transition_linear(
            self.stacked_transitions.permute(2, 1, 0)
        ).permute(2, 1, 0).squeeze(0)

        ensemble_input = torch.stack(input)
        ensemble_input = ensemble_input.detach()
        ensemble_input = self.input_linear(
            ensemble_input.permute(3, 1, 2, 0)
        ).permute(3, 1, 2, 0).squeeze(0)

        preds = self.ensemble_crf.forward(
                ensemble_input, None, seq_len, decode=True
        )

        loss = None
        if tags is not None:
            loss = self.ensemble_crf.forward(
                ensemble_input, tags, seq_len
            )
            alpha = 0.1
            input_linear_norm = alpha * torch.norm(self.input_linear.weight)
            transition_linear_norm = alpha * torch.norm(self.transition_linear.weight)
            loss += input_linear_norm + transition_linear_norm

        return preds, loss

        # all_paths_scores = []
        # for i, crf in enumerate(self.crf_layers):
        #     paths_socres = self.single_forward(char_bi_lstm_out[i], seq_len)
        #     all_paths_scores.append(paths_socres)
        #
        # for i, p in enumerate(all_paths_scores):
        #     normalized_p =


    def single_forward(self, input, seq_len):
        # get batch info
        batch_size, max_seq_len, _ = input.size()

        # shape = [batch_size, seq_len, num_labels]
        pred = self.linear(input)

        # set a min value
        small = -1000
        # set the values of <s> and <e> labels for each token to small
        pred[:, :, -2] = small
        pred[:, :, -1] = small
        # create start and end of the sentence state
        start_state = Variable(
            torch.zeros(batch_size, 1, self.num_labels).fill_(small))
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
            torch.FloatTensor(max_seq_len + 1, batch_size, self.num_labels, self.num_labels)
        )

        if self.transitions.is_cuda:
            paths_scores = paths_scores.cuda()

        previous = padded_pred[0]
        for i in range(1, len(padded_pred)):
            _previous = previous.unsqueeze(2)
            _padded_pred = padded_pred[i].unsqueeze(1)
            # decode
            scores = _previous + _padded_pred + self.transitions
            paths_scores[i - 1] = scores

        return paths_scores