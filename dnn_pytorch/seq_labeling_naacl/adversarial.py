import os
import numpy as np
import scipy
import scipy.linalg

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from muse_utils import clip_parameters, load_embeddings, export_embeddings
from muse_utils import normalize_embeddings
from muse_dico_builder import build_dictionary

from layers import Embedding, CharCnnWordEmb, EncodeLstm, CnnDiscriminator, LinearProj, CRFLoss
from loader import load_pretrained


def build_word_adversarial_model(params):
    # prepare data for word level adversarial
    target_dico, _target_emb = load_embeddings(params, target=True)
    params['target_dico'] = target_dico
    target_emb = nn.Embedding(len(target_dico), params['word_dim'], sparse=True)
    target_emb.weight.data.copy_(_target_emb)

    # target embeddings
    related_dico, _related_emb = load_embeddings(params, target=False)
    params['related_dico'] = related_dico
    related_emb = nn.Embedding(len(related_dico), params['word_dim'], sparse=True)
    related_emb.weight.data.copy_(_related_emb)

    params['target_mean'] = normalize_embeddings(target_emb.weight.data, params['normalize_embeddings'])
    params['related_mean'] = normalize_embeddings(related_emb.weight.data, params['normalize_embeddings'])

    # embedding projection function
    embedding_mapping = EmbeddingMapping(params['word_dim'], params['word_dim'])
    # define the word discriminator
    word_discriminator = WordDiscriminator(input_dim=params['word_dim'],
                                           hidden_dim=params['dis_hid_dim'],
                                           dis_layers=params['dis_layers'],
                                           dis_input_dropout=params['dis_input_dropout'],
                                           dis_dropout=params['dis_dropout'])

    if params['gpu']:
        target_emb = target_emb.cuda()
        related_emb = related_emb.cuda()
        embedding_mapping = embedding_mapping.cuda()
        word_discriminator = word_discriminator.cuda()

    return target_emb, related_emb, embedding_mapping, word_discriminator

class WordAdversarialTrainer(object):
    def __init__(self, target_emb, related_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.target_emb = target_emb
        self.related_emb = related_emb
        self.target_dico = params['target_dico']
        self.related_dico = params['related_dico']
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params

        # optimizers
        map_optim_fn = optim.SGD
        if params['map_optimizer'] == 'adadelta':
            map_optim_fn = optim.Adadelta
        elif params['map_optimizer'] == 'adagrad':
            map_optim_fn = optim.Adagrad
        elif params['map_optimizer'] == 'adam':
            map_optim_fn = optim.Adam
        elif params['map_optimizer'] == 'adamax':
            map_optim_fn = optim.Adamax
        elif params['map_optimizer'] == 'asgd':
            map_optim_fn = optim.ASGD
        elif params['map_optimizer'] == 'rmsprop':
            map_optim_fn = optim.RMSprop
        elif params['map_optimizer'] == 'rprop':
            map_optim_fn = optim.Rprop
        elif params['map_optimizer'] == 'sgd':
            map_optim_fn = optim.SGD

        self.map_optimizer = map_optim_fn(mapping.parameters(), lr=params['map_learning_rate'], momentum=0.9)

        dis_optim_fn = optim.SGD
        if params['dis_optimizer'] == 'adadelta':
            dis_optim_fn = optim.Adadelta
        elif params['dis_optimizer'] == 'adagrad':
            dis_optim_fn = optim.Adagrad
        elif params['dis_optimizer'] == 'adam':
            dis_optim_fn = optim.Adam
        elif params['dis_optimizer'] == 'adamax':
            dis_optim_fn = optim.Adamax
        elif params['dis_optimizer'] == 'asgd':
            dis_optim_fn = optim.ASGD
        elif params['dis_optimizer'] == 'rmsprop':
            dis_optim_fn = optim.RMSprop
        elif params['dis_optimizer'] == 'rprop':
            dis_optim_fn = optim.Rprop
        elif params['dis_optimizer'] == 'sgd':
            dis_optim_fn = optim.SGD

        self.dis_optimizer = dis_optim_fn(discriminator.parameters(), lr=params['dis_learning_rate'], momentum=0.9)

        # best validation score
        self.best_valid_metric = -1e12
        self.decrease_lr = False

    def dis_step(self, target_ids, related_ids):
        self.discriminator.train()

        # get word embeddings
        target_emb_tmp = self.target_emb(Variable(target_ids, volatile=True))
        related_emb_tmp = self.related_emb(Variable(related_ids, volatile=True))
        related_emb_tmp = self.mapping(Variable(related_emb_tmp.data, volatile=True))
        target_emb_tmp = Variable(target_emb_tmp.data, volatile=True)

        batch_size = target_ids.size(0)

        # input / target
        x = torch.cat([related_emb_tmp, target_emb_tmp], 0)
        y = torch.FloatTensor(batch_size * 2).zero_()
        y[:batch_size] = 1 - self.params['dis_smooth']
        y[batch_size:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)

        # loss
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, 0) # self.params['dis_clip_weights']
        return loss

    def mapping_step(self, target_ids, related_ids):
        """
        Fooling discriminator training step.
        """
        if self.params['dis_lambda'] == 0:
            return 0

        self.discriminator.eval()

        # get word embeddings
        target_emb_tmp = self.target_emb(Variable(target_ids, volatile=True))
        related_emb_tmp = self.related_emb(Variable(related_ids, volatile=True))
        related_emb_tmp = self.mapping(Variable(related_emb_tmp.data, volatile=True))
        target_emb_tmp = Variable(target_emb_tmp.data, volatile=True)

        batch_size = target_ids.size(0)

        # input / target
        x = torch.cat([related_emb_tmp, target_emb_tmp], 0)
        y = torch.FloatTensor(batch_size * 2).zero_()
        y[:batch_size] = 1 - self.params['dis_smooth']
        y[batch_size:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)

        # loss
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params['dis_lambda'] * loss

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params['batch_size'], loss

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params['map_beta'] > 0:
            W = self.mapping.mapper.weight.data
            beta = self.params['map_beta']
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params['map_optimizer']:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params['min_lr'], old_lr * self.params['lr_decay'])
        if new_lr < old_lr:
            print("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                print("Validation metric is smaller than the best: %.5f vs %.5f"
                      % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params['lr_shrink']
                    print("Shrinking the learning rate: %.5f -> %.5f"
                          % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            print('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.mapper.weight.data.cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            print('* Saving the mapping to %s ...' % path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params['exp_path'], 'best_mapping.pth')
        print('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.mapper.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        print("Reloading all embeddings for mapping ...")
        params['target_dico'], target_emb = load_embeddings(params, target=True, full_vocab=True)
        params['related_dico'], related_emb = load_embeddings(params, target=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(target_emb, params['normalize_embeddings'], mean=params['target_mean'])
        normalize_embeddings(related_emb, params['normalize_embeddings'], mean=params['related_mean'])

        # map source embeddings to the target space
        bs = 4096
        print("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(related_emb), bs)):
            x = Variable(related_emb[k:k + bs], volatile=True)
            related_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(target_emb, related_emb, params)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        related_emb = self.mapping(self.related_emb.weight).data
        target_emb = self.target_emb.weight.data
        related_emb = related_emb / related_emb.norm(2, 1, keepdim=True).expand_as(related_emb)
        target_emb = target_emb / target_emb.norm(2, 1, keepdim=True).expand_as(target_emb)
        self.dico = build_dictionary(related_emb, target_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.related_emb.weight.data[self.dico[:, 0]]
        B = self.target_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.mapper.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


def build_sequence_adversarial_model(params, mappings):
    # construct word embedding layers
    target_word_embedding = Embedding(params['word_vocab_size'], params['word_dim'])
    related_word_embedding = Embedding(params['bi_word_vocab_size'], params['word_dim'])
    load_pretrained(target_word_embedding.emb, mappings['id_to_word'], params['pre_emb'])
    load_pretrained(related_word_embedding.emb, mappings['bi_id_to_word'], params['bi_pre_emb'])

    # char embedding layer
    char_embedding = Embedding(params['char_vocab_size'], params['char_dim'])

    # CNN and concatenate with word for target language
    target_char_cnn_word = CharCnnWordEmb(params['word_dim'], params['char_dim'], params['char_conv'],
                                          params['max_word_length'], params['filter_withs'])

    # CNN and concatenate with word for related language
    related_char_cnn_word = CharCnnWordEmb(params['word_dim'], params['char_dim'], params['char_conv'],
                                           params['max_word_length'], params['filter_withs'])

    # sequence encoder
    adv_lstm = EncodeLstm(params['char_cnn_word_dim'], params['char_cnn_word_dim'], bidrection=True,
                          dropout=params['dropout'])

    # sequence discriminator
    seq_discriminator = CnnDiscriminator(params['char_cnn_word_dim']*2, params['word_lstm_dim'], [2, 3], 1)

    # context encoder
    context_lstm = EncodeLstm(params['char_cnn_word_dim']*2, params['word_lstm_dim'],
                              dropout=params['dropout'])
    # linear projection
    linear_proj = LinearProj(params['word_lstm_dim'] * 2, params['word_lstm_dim'], params['label_size'])

    tagger_criterion = CRFLoss(params['label_size'])

    dis_criterion = nn.NLLLoss()

    if params['gpu']:
        target_word_embedding = target_word_embedding.cuda()
        related_word_embedding = related_word_embedding.cuda()
        char_embedding = char_embedding.cuda()
        target_char_cnn_word = target_char_cnn_word.cuda()
        related_char_cnn_word = related_char_cnn_word.cuda()
        adv_lstm = adv_lstm.cuda()
        seq_discriminator = seq_discriminator.cuda()
        context_lstm = context_lstm.cuda()
        linear_proj = linear_proj.cuda()
        tagger_criterion = tagger_criterion.cuda()
        dis_criterion = dis_criterion.cuda()

    return target_word_embedding, related_word_embedding, char_embedding, target_char_cnn_word, \
           related_char_cnn_word, adv_lstm, seq_discriminator, context_lstm, linear_proj, \
           tagger_criterion, dis_criterion


class SeqAdversarialTrainer(object):
    def __init__(self, target_word_embedding, related_word_embedding, embedding_mapping, char_embedding,
                 target_char_cnn_word, related_char_cnn_word, adv_lstm, seq_discriminator, context_lstm,
                 linear_proj, tagger_criterion, dis_criterion, params):
        """
        Initialize trainer script.
        """
        self.target_word_embedding = target_word_embedding
        self.related_word_embedding = related_word_embedding
        self.embedding_mapping = embedding_mapping
        self.char_embedding = char_embedding
        self.target_char_cnn_word = target_char_cnn_word
        self.related_char_cnn_word = related_char_cnn_word
        self.adv_lstm = adv_lstm
        self.seq_discriminator = seq_discriminator
        self.context_lstm = context_lstm
        self.linear_proj = linear_proj
        self.tagger_criterion = tagger_criterion
        self.dis_criterion = dis_criterion
        self.params = params

        feature_parameters = []
        feature_parameters += char_embedding.parameters()
        feature_parameters += target_char_cnn_word.parameters()
        feature_parameters += related_char_cnn_word.parameters()
        feature_parameters += adv_lstm.parameters()
        feature_optim = optim.SGD
        if params['tagger_optimizer'] == 'adadelta':
            feature_optim = optim.Adadelta
        elif params['tagger_optimizer'] == 'adagrad':
            feature_optim = optim.Adagrad
        elif params['tagger_optimizer'] == 'adam':
            feature_optim = optim.Adam
        elif params['tagger_optimizer'] == 'adamax':
            feature_optim = optim.Adamax
        elif params['tagger_optimizer'] == 'asgd':
            feature_optim = optim.ASGD
        elif params['tagger_optimizer'] == 'rmsprop':
            feature_optim = optim.RMSprop
        elif params['tagger_optimizer'] == 'rprop':
            feature_optim = optim.Rprop
        elif params['tagger_optimizer'] == 'sgd':
            feature_optim = optim.SGD
        self.feature_parameters = feature_parameters
        self.feature_optimizer = feature_optim(feature_parameters, lr=params['tagger_learning_rate'], momentum=0.9)

        tagger_parameters = []
        tagger_parameters += context_lstm.parameters()
        tagger_parameters += linear_proj.parameters()
        tagger_parameters += tagger_criterion.parameters()
        # optimizers
        tagger_optim = optim.SGD
        if params['tagger_optimizer'] == 'adadelta':
            tagger_optim = optim.Adadelta
        elif params['tagger_optimizer'] == 'adagrad':
            tagger_optim = optim.Adagrad
        elif params['tagger_optimizer'] == 'adam':
            tagger_optim = optim.Adam
        elif params['tagger_optimizer'] == 'adamax':
            tagger_optim = optim.Adamax
        elif params['tagger_optimizer'] == 'asgd':
            tagger_optim = optim.ASGD
        elif params['tagger_optimizer'] == 'rmsprop':
            tagger_optim = optim.RMSprop
        elif params['tagger_optimizer'] == 'rprop':
            tagger_optim = optim.Rprop
        elif params['tagger_optimizer'] == 'sgd':
            tagger_optim = optim.SGD
        self.tagger_parameters = tagger_parameters
        self.tagger_optimizer = tagger_optim(tagger_parameters, lr=params['tagger_learning_rate'], momentum=0.9)

        discriminator_parameters = []
        discriminator_parameters += seq_discriminator.parameters()
        # optimizers
        discriminator_optim = optim.SGD
        self.discriminator_parameters = discriminator_parameters
        self.discriminator_optimizer = discriminator_optim(discriminator_parameters, lr=params['tagger_learning_rate'],
                                                           momentum=0.9)

    def pretrain_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.eval()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.eval()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        loss_tagger = target_tagger_loss

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.feature_optimizer.zero_grad()
        self.tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        self.feature_optimizer.step()
        self.tagger_optimizer.step()
        return loss_tagger

    # only seq_discriminator and dis_criterion are trainable
    # only optimize dis loss
    def dis_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, related_word_ids,
                 related_char_ids, related_char_len, related_seq_len):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.train()
        self.context_lstm.eval()
        self.linear_proj.eval()
        self.tagger_criterion.eval()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)
        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        # discriminator
        batch_size = target_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        y[:] = 1 - self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        target_discriminator_output = self.seq_discriminator.forward(target_adv_lstm_output)

        loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
        loss_dis_target /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        batch_size = related_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        y[:] = self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)

        loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
        loss_dis_related /= len(related_seq_len)

        loss_dis = loss_dis_target + loss_dis_related

        # check NaN
        if (loss_dis != loss_dis).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.feature_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.discriminator_parameters, 5)
        self.feature_optimizer.step()
        self.discriminator_optimizer.step()
        # clip_parameters(self.dis_seq_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_dis

    def dis_step1(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, related_word_ids,
                 related_char_ids, related_char_len, related_seq_len):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.train()
        self.context_lstm.eval()
        self.linear_proj.eval()
        self.tagger_criterion.eval()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)
        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        # discriminator
        batch_size = target_word_ids.size(0)
        s = np.random.normal(0.55, 0.1, batch_size)
        y = torch.from_numpy(s).float()
        #y = torch.FloatTensor(batch_size).zero_()
        #y[:] = 1 - self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        target_discriminator_output = self.seq_discriminator.forward(target_adv_lstm_output)

        loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
        loss_dis_target /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        batch_size = related_word_ids.size(0)
        s = 1 - np.random.normal(0.55, 0.1, batch_size)
        y = torch.from_numpy(s).float()
        # y = torch.FloatTensor(batch_size).zero_()
        # y[:] = self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)

        loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
        loss_dis_related /= len(related_seq_len)

        loss_dis = loss_dis_target + loss_dis_related

        # check NaN
        if (loss_dis != loss_dis).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.feature_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.discriminator_parameters, 5)
        self.feature_optimizer.step()
        self.discriminator_optimizer.step()
        # clip_parameters(self.dis_seq_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_dis

    def tagger_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
                    related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        # discriminator
        batch_size = target_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        y[:] = 1 #- self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        target_discriminator_output = self.seq_discriminator.forward(target_adv_lstm_output)

        loss_dis_target = F.binary_cross_entropy(target_discriminator_output, 1 - y)
        loss_dis_target /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)

        batch_size = related_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        #y[:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)

        loss_dis_related = F.binary_cross_entropy(related_discriminator_output, 1 - y)
        loss_dis_related /= len(related_seq_len)

        related_discriminator_output_target = 1 - related_discriminator_output
        related_discriminator_output_all = torch.stack([related_discriminator_output_target,
                                                        related_discriminator_output], 1)
        related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
        related_discriminator_output_label = related_discriminator_output_label.float()

        related_tagger_loss = self.tagger_criterion.forward(related_pred_probs, related_reference, related_seq_len,
                                                            decode=False)
        related_tagger_loss /= len(related_seq_len)

        # loss_dis = loss_dis_target + loss_dis_related
        loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis_target + loss_dis_related #loss_dis

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.feature_optimizer.zero_grad()
        self.tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        self.feature_optimizer.step()
        self.tagger_optimizer.step()
        # clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_tagger

    def tagger_step1(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
                    related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        # discriminator
        batch_size = target_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        y[:] = 1 - self.params['seq_dis_smooth']-0.3
        y = Variable(y.cuda() if self.params['gpu'] else y)
        target_discriminator_output = self.seq_discriminator.forward(target_adv_lstm_output)

        loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
        loss_dis_target /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)

        batch_size = related_word_ids.size(0)
        y = torch.FloatTensor(batch_size).zero_()
        y[:] = self.params['seq_dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)

        loss_dis_related = F.binary_cross_entropy(related_discriminator_output, 1 - y)
        loss_dis_related /= len(related_seq_len)

        related_discriminator_output_target = 1 - related_discriminator_output
        related_discriminator_output_all = torch.stack([related_discriminator_output_target,
                                                        related_discriminator_output], 1)
        related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
        related_discriminator_output_label = related_discriminator_output_label.float()

        related_tagger_loss = self.tagger_criterion.forward(related_pred_probs, related_reference, related_seq_len,
                                                            decode=False, batch_mask=related_discriminator_output_label)
        related_tagger_loss /= len(related_seq_len)

        # loss_dis = loss_dis_target + loss_dis_related
        loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis_target + loss_dis_related #loss_dis

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.feature_optimizer.zero_grad()
        self.tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        self.feature_optimizer.step()
        self.tagger_optimizer.step()
        # clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_tagger

    def tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference,
                           related_word_ids, related_char_ids, related_char_len, related_seq_len, related_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        # discriminator
        batch_size = target_word_ids.size(0)
        y = torch.LongTensor(batch_size).zero_()
        y[:] = self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        target_discriminator_output = self.seq_discriminator.forward(target_adv_lstm_output)
        # loss_dis_target = F.binary_cross_entropy(target_discriminator_output, y)
        # loss_dis_target = self.dis_criterion(target_discriminator_output, y)
        loss_dis_target = F.cross_entropy(target_discriminator_output, 1 - y)
        loss_dis_target /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)
        related_tagger_loss = self.tagger_criterion.forward(related_pred_probs, related_reference, related_seq_len,
                                                            decode=False)
        related_tagger_loss /= len(related_seq_len)

        batch_size = related_word_ids.size(0)
        y = torch.LongTensor(batch_size).zero_()
        y[:] = 1 - self.params['dis_smooth']
        y = Variable(y.cuda() if self.params['gpu'] else y)
        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)
        # loss_dis_related = self.dis_criterion(related_discriminator_output, y)
        # loss_dis_related = F.binary_cross_entropy(related_discriminator_output, y)
        loss_dis_related = F.cross_entropy(related_discriminator_output, 1 - y)
        loss_dis_related /= len(related_seq_len)

        loss_dis = loss_dis_target + loss_dis_related
        loss_tagger = target_tagger_loss + related_tagger_loss + loss_dis

        # check NaN
        if (loss_dis != loss_dis).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        self.tagger_optimizer.step()
        # clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_tagger

    def target_tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len,
                                  target_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.eval()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.eval()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        loss_tagger = target_tagger_loss

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.mono_tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.mono_tagger_parameters, 5)
        self.mono_tagger_optimizer.step()

        return loss_tagger

    def related_tagging_train_step(self, related_word_ids, related_char_ids, related_char_len, related_seq_len,
                                   related_reference):

        self.target_word_embedding.eval()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.eval()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)
        related_tagger_loss = self.tagger_criterion.forward(related_pred_probs, related_reference, related_seq_len,
                                                            decode=False)
        related_tagger_loss /= len(related_seq_len)
        loss_tagger = related_tagger_loss

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.related_tagger_optimizer.zero_grad()
        loss_tagger.backward()
        torch.nn.utils.clip_grad_norm(self.related_tagger_parameters, 5)
        self.related_tagger_optimizer.step()
        # clip_parameters(self.tagger_optimizer, 0)  # self.params['dis_clip_weights']

        return loss_tagger

    def combine_tagging_train_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len,
                                   target_reference, related_word_ids, related_char_ids, related_char_len,
                                   related_seq_len, related_reference):

        self.target_word_embedding.train()
        self.related_word_embedding.train()
        self.embedding_mapping.eval()
        self.char_embedding.train()
        self.target_char_cnn_word.train()
        self.related_char_cnn_word.train()
        self.adv_lstm.train()
        self.seq_discriminator.eval()
        self.context_lstm.train()
        self.linear_proj.train()
        self.tagger_criterion.train()
        self.dis_criterion.train()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        target_tagger_loss = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len,
                                                           decode=False)
        target_tagger_loss /= len(target_seq_len)

        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)

        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)
        # related_max, related_discriminator_output_label = torch.max(related_discriminator_output, 1)
        # related_discriminator_output_label = related_discriminator_output_label.float()

        related_discriminator_output_neg = 1 - related_discriminator_output
        related_discriminator_output_all = torch.stack([related_discriminator_output, related_discriminator_output_neg],
                                                       1)
        related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
        related_discriminator_output_label = related_discriminator_output_label.float()

        related_tagger_loss = self.tagger_criterion.forward(related_pred_probs, related_reference, related_seq_len,
                                                            decode=False) # , batch_mask=related_discriminator_output_label
        related_tagger_loss /= len(related_seq_len)

        loss_tagger = target_tagger_loss + related_tagger_loss

        # check NaN
        if (loss_tagger != loss_tagger).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        # self.feature_optimizer.zero_grad()
        # self.tagger_optimizer.zero_grad()
        # loss_tagger.backward()
        # torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        # torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        # self.feature_optimizer.step()
        # self.tagger_optimizer.step()

        #self.feature_optimizer.zero_grad()
        self.tagger_optimizer.zero_grad()
        loss_tagger.backward()
        #torch.nn.utils.clip_grad_norm(self.feature_parameters, 5)
        torch.nn.utils.clip_grad_norm(self.tagger_parameters, 5)
        #self.feature_optimizer.step()
        self.tagger_optimizer.step()

        return loss_tagger

    def tagging_dev_step(self, target_word_ids, target_char_ids, target_char_len, target_seq_len, target_reference):

        self.target_word_embedding.eval()
        self.related_word_embedding.eval()
        self.embedding_mapping.eval()
        self.char_embedding.eval()
        self.target_char_cnn_word.eval()
        self.related_char_cnn_word.eval()
        self.adv_lstm.eval()
        self.seq_discriminator.eval()
        self.context_lstm.eval()
        self.linear_proj.eval()
        self.tagger_criterion.eval()
        self.dis_criterion.eval()

        # batchsize * seq_length * word_emb
        input_target_words = self.target_word_embedding.forward(target_word_ids)
        # batchsize * seq_length * max_char_length * char_emb
        input_target_chars = self.char_embedding.forward(target_char_ids)
        target_combined_word_input, target_combined_word_input_dim = \
            self.target_char_cnn_word.forward(input_target_chars, target_char_len, input_target_words)

        target_adv_lstm_output = self.adv_lstm.forward(target_combined_word_input, target_seq_len, len(target_seq_len),
                                                       dropout=self.params['dropout'])

        target_context_lstm_output = self.context_lstm.forward(target_adv_lstm_output, target_seq_len,
                                                               len(target_seq_len), dropout=self.params['dropout'])
        target_pred_probs = self.linear_proj.forward(target_context_lstm_output)

        dev_pred_seq = self.tagger_criterion.forward(target_pred_probs, target_reference, target_seq_len, decode=True)

        return dev_pred_seq

    def related_tagging_dev_step(self, related_word_ids, related_char_ids, related_char_len, related_seq_len):

        self.target_word_embedding.eval()
        self.related_word_embedding.eval()
        self.embedding_mapping.eval()
        self.char_embedding.eval()
        self.target_char_cnn_word.eval()
        self.related_char_cnn_word.eval()
        self.adv_lstm.eval()
        self.seq_discriminator.eval()
        self.context_lstm.eval()
        self.linear_proj.eval()
        self.tagger_criterion.eval()
        self.dis_criterion.eval()

        # for related languages
        # for related languages
        input_related_words_old = self.related_word_embedding.forward(related_word_ids)
        input_related_words = self.embedding_mapping.forward(input_related_words_old)
        input_related_chars = self.char_embedding.forward(related_char_ids)
        related_combined_word_input, related_combined_word_input_dim = \
            self.related_char_cnn_word.forward(input_related_chars, related_char_len, input_related_words)

        related_adv_lstm_output = self.adv_lstm.forward(related_combined_word_input, related_seq_len,
                                                        len(related_seq_len), dropout=self.params['dropout'])

        related_context_lstm_output = self.context_lstm.forward(related_adv_lstm_output, related_seq_len,
                                                                len(related_seq_len), dropout=self.params['dropout'])
        related_pred_probs = self.linear_proj.forward(related_context_lstm_output)

        related_discriminator_output = self.seq_discriminator.forward(related_adv_lstm_output)
        related_discriminator_output_target = 1 - related_discriminator_output
        related_discriminator_output_all = torch.stack([related_discriminator_output_target,
                                                        related_discriminator_output], 1)
        related_max, related_discriminator_output_label = torch.max(related_discriminator_output_all, 1)
        related_discriminator_output_label = related_discriminator_output_label.float()

        return related_max, related_discriminator_output_label, related_discriminator_output


class EmbeddingMapping(nn.Module):
    def __init__(self, mono_dim, common_dim):
        super(EmbeddingMapping, self).__init__()
        self.mono_dim = mono_dim
        self.common_dim = common_dim
        self.mapper = nn.Linear(mono_dim, common_dim, bias=False)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, input):
        encoded = self.mapper(input)
        return encoded

    def orthogonalize(self, map_beta):
        """
        Orthogonalize the mapping.
        """
        if map_beta > 0:
            W = self.mapper.weight.data
            beta = map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


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


class WordDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dis_layers, dis_input_dropout, dis_dropout):
        """Init discriminator."""
        super(WordDiscriminator, self).__init__()

        self.emb_dim = input_dim
        self.dis_layers = dis_layers
        self.dis_hid_dim = hidden_dim
        self.dis_dropout = dis_dropout
        self.dis_input_dropout = dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]

        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
            else:
                nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)








