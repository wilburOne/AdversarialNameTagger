import argparse
import itertools
import os
import random
import sys
import time
from parse_args import parse_args

import torch
import torch.nn as nn

from evaluate import evaluate_ner
from muse_utils import bool_flag
from tag_scheme import update_tag_scheme
from muse_evaluator import all_eval
from loader import augment_with_pretrained, augment_with_pretrained_bi, load_sentences, load_features
from preprocessing import Tee, create_input, prepare_dataset, prepare_mapping_bi, prepare_dataset_bi, to_var
from adversarial import build_word_adversarial_model, WordAdversarialTrainer, \
    build_sequence_adversarial_model, SeqAdversarialTrainer


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'

# Read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument("--train", default="", help="Train set location")
parser.add_argument("--bi_train", default="", help="Bi_Train set location")
parser.add_argument("--dev", default="", help="Dev set location")
parser.add_argument("--test", default="", help="Test set location")
parser.add_argument("--model_dp", default="", help="model directory path")
parser.add_argument("--tag_scheme", default="iobes", help="Tagging scheme (IOB or IOBES)")
parser.add_argument("--lower", default='0', type=int, help="Lowercase words (this will not affect character inputs)")
parser.add_argument("--zeros", default="1", type=int, help="Replace digits with 0")
parser.add_argument("--char_dim", default="25", type=int, help="Char embedding dimension")
parser.add_argument("--char_lstm_dim", default="25", type=int, help="Char LSTM hidden layer size")
parser.add_argument("--char_cnn", default="1", type=int, help="Use CNN to generate char embeddings.(0 to disable)")
parser.add_argument("--char_conv", default="25", type=int, help="filter number")
parser.add_argument("--word_dim", default="100", type=int, help="Token embedding dimension")
parser.add_argument("--word_lstm_dim", default="100", type=int, help="Token LSTM hidden layer size")
parser.add_argument("--pre_emb", default="", help="Location of pretrained embeddings")
parser.add_argument("--bi_pre_emb", default="", help="Location of bi_pretrained embeddings")
parser.add_argument("--all_emb", default="0", type=int, help="Load all embeddings")
parser.add_argument("--feat", default="0", type=int, help="file path of external features.")
parser.add_argument("--crf", default="1", type=int, help="Use CRF (0 to disable)")
parser.add_argument("--dropout", default="0.5", type=float, help="Droupout on the input (0 = no dropout)")
parser.add_argument("--tagger_learning_rate", default="0.01", type=float, help="learning rate for the tagger")
parser.add_argument("--tagger_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
parser.add_argument("--dis_seq_learning_rate", default="0.01", type=float, help="learning rate for the discriminator")
parser.add_argument("--dis_seq_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
parser.add_argument("--mapping_seq_learning_rate", default="0.01", type=float, help="learning rate for the mapper")
parser.add_argument("--mapping_seq_optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
parser.add_argument("--lr_method", default="sgd-lr_.005", help="Learning method (SGD, Adadelta, Adam..)")
parser.add_argument("--optimizer", default="sgd", help="Learning method (SGD, Adadelta, Adam..)")
parser.add_argument("--num_epochs", default="100", type=int, help="Number of training epochs")
parser.add_argument("--batch_size", default="20", type=int, help="Batch size.")
parser.add_argument("--gpu", default="1", type=int, help="default is 0. set 1 to use gpu.")
parser.add_argument("--cuda", default="1", type=int, help="gpu number.")
parser.add_argument("--signal", default="", type=str)
# parameters for word adversarial training
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--target_lang", type=str, default='en', help="target language")
parser.add_argument("--related_lang", type=str, default='es', help="related language")
parser.add_argument("--max_vocab", type=int, default=500000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=100000, help="Select embeddings of the k most frequent "
                                                                          "words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--adv_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--adv_iteration", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--adv_batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_learning_rate", type=float, default=0.1, help="learning rate")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_learning_rate", type=float, default=0.1, help="learning rate")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation "
                                                                 "metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the "
                                                                "refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation "
                                                                           "(nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--target_emb", type=str, default="", help="target embeddings")
parser.add_argument("--related_emb", type=str, default="", help="related embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")

args = parser.parse_args()
params, model_names = parse_args(args)

params['seq_dis_smooth'] = 0.3

torch.cuda.set_device(args.cuda)

# generate model name
model_dir = args.model_dp
model_name = []
for k in model_names:
    v = params[k]
    if not v:
        continue
    if k == 'pre_emb':
        v = os.path.basename(v)
    model_name.append('='.join((k, str(v))))
model_dir = os.path.join(model_dir, ','.join(model_name[:-1]))
os.makedirs(model_dir, exist_ok=True)
params['model_dp'] = model_dir

# register logger to save print(messages to both stdout and disk)
training_log_path = os.path.join(model_dir, 'training_log.txt')
if os.path.exists(training_log_path):
    os.remove(training_log_path)
f = open(training_log_path, 'w')
sys.stdout = Tee(sys.stdout, f)

# print model parameters
print('Training data: %s' % params['train'])
print('Bi_Training data: %s' % params['bi_train'])
print('Dev data: %s' % params['dev'])
print('Test data: %s' % params['test'])
print("Model location: %s" % params['model_dp'])
print("Target embedding: %s " % params['target_emb'])
print("Related embedding: %s " % params['related_emb'])

eval_path = os.path.join(os.path.dirname(__file__), "./evaluation")
eval_script = os.path.join(eval_path, 'conlleval')
params['eval_script'] = eval_script

# Data parameters
lower = params['lower']
zeros = params['zeros']
tag_scheme = params['tag_scheme']
params['max_word_length'] = 25
params['filter_withs'] = [2, 3, 4]
params['char_cnn_word_dim'] = params['word_dim'] + params['char_conv'] * len(params['filter_withs'])
params['shared_lstm_hidden_dim'] = params['word_lstm_dim']

# Load sentences
train_sentences = load_sentences(params['train'], to_sort=False)
bi_train_sentences = load_sentences(params['bi_train'], to_sort=False)
dev_sentences = load_sentences(params['dev'])
test_sentences = load_sentences(params['test'])

# Use selected tagging scheme (IOB / IOBES), also check tagging scheme
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(bi_train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# load external features if provided
features = []
if params['feat']:
    features = load_features(params['feat'])

# prepare mappings
all_sentences = train_sentences + dev_sentences + test_sentences
mappings = prepare_mapping_bi(all_sentences, bi_train_sentences, features, **params)

# If pretrained embeddings is used and all_emb flag is on,
# we augment the words by pretrained embeddings.
# if parameters['pre_emb'] and parameters['all_emb']:
updated_word_mappings = augment_with_pretrained(all_sentences, params['pre_emb'])
mappings.update(updated_word_mappings)

updated_word_mappings = augment_with_pretrained_bi(bi_train_sentences, params['bi_pre_emb'])
mappings.update(updated_word_mappings)

# compute vocab size
params['label_size'] = len(mappings['id_to_tag'])
params['word_vocab_size'] = len(mappings['id_to_word'])
params['bi_word_vocab_size'] = len(mappings['bi_id_to_word'])
params['char_vocab_size'] = len(mappings['id_to_char'])
params['feat_vocab_size'] = [len(item) for item in mappings['id_to_feat_list']]

print("word vocab size: ", params['word_vocab_size'])
print("bi word vocab size: ", params['bi_word_vocab_size'])

# Index data
dataset = dict()
dataset['train'] = prepare_dataset(train_sentences, mappings, zeros, lower, is_train=True)
dataset['bi_train'] = prepare_dataset_bi(bi_train_sentences, mappings, zeros, lower, is_train=True)
dataset['dev'] = prepare_dataset(dev_sentences, mappings, zeros, lower, is_train=True)
dataset['test'] = prepare_dataset(test_sentences, mappings, zeros, lower, is_train=True)

print("%i / %i / %i / %i sentences in train / bi-train / dev / test." % (
    len(dataset['train']), len(dataset['bi_train']), len(dataset['dev']), len(dataset['test'])))

# initialize model
print('model initializing...')

target_emb, related_emb, embedding_mapping, word_discriminator = build_word_adversarial_model(params)
adv_trainer = WordAdversarialTrainer(target_emb, related_emb, embedding_mapping, word_discriminator, params)

target_word_embedding, related_word_embedding, char_embedding, target_char_cnn_word, related_char_cnn_word, \
adv_lstm, seq_discriminator, context_lstm, linear_proj, tagger_criterion, dis_criterion = \
    build_sequence_adversarial_model(params, mappings)

seq_trainer = SeqAdversarialTrainer(target_word_embedding, related_word_embedding, embedding_mapping, char_embedding,
                                           target_char_cnn_word, related_char_cnn_word, adv_lstm, seq_discriminator,
                                           context_lstm, linear_proj, tagger_criterion, dis_criterion, params)

module_list = nn.ModuleList([target_emb, related_emb, embedding_mapping, word_discriminator, target_word_embedding,
                             related_word_embedding, char_embedding, target_char_cnn_word, related_char_cnn_word,
                             adv_lstm, seq_discriminator, context_lstm, linear_proj, tagger_criterion, dis_criterion])

print('----> WORD ADVERSARIAL TRAINING <----\n')
best_valid_metric = 0
for n_epoch in range(params['adv_epochs']):
    print('Starting adversarial training epoch %i...' % n_epoch)
    for n_iter in range(0, params['adv_iteration'], params['adv_batch_size']):
        # discriminator training
        dis_loss = 0
        for _ in range(params['dis_steps']):
            # select random word IDs
            target_ids = torch.LongTensor(params['adv_batch_size']).random_(len(params['target_dico'])
                                                                            if params['dis_most_frequent'] == 0
                                                                            else params['dis_most_frequent'])
            related_ids = torch.LongTensor(params['adv_batch_size']).random_(len(params['related_dico'])
                                                                             if params['dis_most_frequent'] == 0
                                                                             else params['dis_most_frequent'])

            if params['gpu']:
                target_ids = target_ids.cuda()  # pre_emb
                related_ids = related_ids.cuda()
            dis_loss += adv_trainer.dis_step(target_ids, related_ids)
        dis_loss /= params['dis_steps']

        # mapping training (discriminator fooling)
        # select random word IDs
        target_ids = torch.LongTensor(params['adv_batch_size']).random_(len(params['target_dico'])
                                                                        if params['dis_most_frequent'] == 0
                                                                        else params['dis_most_frequent'])
        related_ids = torch.LongTensor(params['adv_batch_size']).random_(len(params['related_dico'])
                                                                         if params['dis_most_frequent'] == 0
                                                                         else params['dis_most_frequent'])

        if params['gpu']:
            target_ids = target_ids.cuda()  # pre_emb
            related_ids = related_ids.cuda()
        p, map_loss = adv_trainer.mapping_step(target_ids, related_ids)

        # sys.stdout.write(
        #     'epoch %i, iter %i, discriminator loss: %f, mapping loss: %f\r' % (
        #         n_epoch, n_iter, dis_loss, map_loss))
        # sys.stdout.flush()
        if n_iter % 10000 == 0:
            print('epoch %i, iter %i, discriminator loss: %f, mapping loss: %f\r' %
                  (n_epoch, n_iter, dis_loss, map_loss))

    # embeddings / discriminator evaluation
    projected_related_emb = embedding_mapping.forward(related_emb.weight)
    valid_metric = all_eval(params['related_dico'], params['target_dico'], projected_related_emb, target_emb,
                            params['dico_eval'], VALIDATION_METRIC)

    if valid_metric > best_valid_metric:
        path = os.path.join(params['model_dp'], 'best_mapping.pth')
        print('* Saving the mapping to %s ...' % path)
        state = {'state_dict': embedding_mapping.state_dict()}
        torch.save(state, path)
        best_valid_metric = valid_metric
    print('End of epoch %i.\n' % n_epoch)

if params['n_refinement'] > 0:
    # training loop
    for n_iter in range(params['n_refinement']):
        print('Starting refinement iteration %i...' % n_iter)
        # build a dictionary from aligned embeddings
        adv_trainer.build_dictionary()
        # apply the Procrustes solution
        adv_trainer.procrustes()
        # embeddings evaluation
        projected_related_emb = embedding_mapping.forward(related_emb.weight)
        valid_metric = all_eval(params['related_dico'], params['target_dico'], projected_related_emb, target_emb,
                                params['dico_eval'], VALIDATION_METRIC)

        # JSON log / save best model / end of epoch
        if valid_metric > best_valid_metric:
            path = os.path.join(params['model_dp'], 'best_mapping.pth')
            state = {'state_dict': embedding_mapping.state_dict()}
            torch.save(state, path)
            print('End of refinement iteration %i.\n' % n_iter)


# training starts
since = time.time()
best_dev = 0.0
best_test = 0.0
best_test_global = 1.0
metric = 'f1'  # use metric 'f1' or 'acc'
num_epochs = params['num_epochs']
batch_size = params['batch_size']
current_patience = 0
patience = 50
num_batches = 0

print('----> SEQUENCE ADVERSARIAL TRAINING <----\n')
for epoch in range(60):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    time_epoch_start = time.time()  # epoch start time

    # training
    target_train_batches = [dataset['train'][i:i + batch_size] for i in range(0, len(dataset['train']), batch_size)]
    related_train_batches_tmp = [dataset['bi_train'][i:i + batch_size] for i in
                                 range(0, len(dataset['bi_train']), batch_size)]
    dev_batches = [dataset['dev'][i:i + batch_size] for i in
                   range(0, len(dataset['dev']), batch_size)]
    test_batches = [dataset['test'][i:i + batch_size] for i in
                    range(0, len(dataset['test']), batch_size)]

    if len(related_train_batches_tmp) < len(target_train_batches):
        related_train_batches = []
        while len(related_train_batches) < len(target_train_batches):
            related_train_batches += related_train_batches_tmp
    else:
        related_train_batches = random.sample(related_train_batches_tmp, len(target_train_batches))

    random.shuffle(target_train_batches)
    random.shuffle(related_train_batches)
    for target_batch, related_batch in zip(target_train_batches, related_train_batches):
        num_batches += 1
        target_inputs = create_input(target_batch, is_cuda=params['gpu'])
        target_word_ids = target_inputs['words']
        target_char_ids = target_inputs['chars']
        target_char_len = target_inputs['char_len']
        target_seq_len = target_inputs['seq_len']
        target_reference = target_inputs['tags']

        related_inputs = create_input(related_batch, is_cuda=params['gpu'])
        related_word_ids = related_inputs['words']
        related_char_ids = related_inputs['chars']
        related_char_len = related_inputs['char_len']
        related_seq_len = related_inputs['seq_len']
        related_reference = related_inputs['tags']

        if torch.cuda.is_available():
            target_word_ids = to_var(target_word_ids)
            target_char_ids = to_var(target_char_ids)
            related_word_ids = to_var(related_word_ids)
            related_char_ids = to_var(related_char_ids)

        dis_loss = seq_trainer.dis_step(target_word_ids, target_char_ids, target_char_len, target_seq_len,
                                        related_word_ids, related_char_ids, related_char_len, related_seq_len)

        map_loss = seq_trainer.tagger_step(target_word_ids, target_char_ids, target_char_len, target_seq_len,
                                           target_reference, related_word_ids, related_char_ids,
                                           related_char_len, related_seq_len, related_reference)

        for _ in range(5):
            rand_idx_related = random.randint(0, len(related_train_batches) - 1)
            rand_idx_target = random.randint(0, len(target_train_batches) - 1)
            target_batch_dis = target_train_batches[rand_idx_target]
            related_batch_dis = related_train_batches[rand_idx_related]

            target_inputs_dis = create_input(target_batch_dis, is_cuda=params['gpu'])
            target_word_ids_dis = target_inputs_dis['words']
            target_char_ids_dis = target_inputs_dis['chars']
            target_char_len_dis = target_inputs_dis['char_len']
            target_seq_len_dis = target_inputs_dis['seq_len']
            target_reference_dis = target_inputs_dis['tags']

            related_inputs_dis = create_input(related_batch_dis, is_cuda=params['gpu'])
            related_word_ids_dis = related_inputs_dis['words']
            related_char_ids_dis = related_inputs_dis['chars']
            related_char_len_dis = related_inputs_dis['char_len']
            related_seq_len_dis = related_inputs_dis['seq_len']
            related_reference_dis = related_inputs_dis['tags']

            if torch.cuda.is_available():
                target_word_ids_dis = to_var(target_word_ids_dis)
                target_char_ids_dis = to_var(target_char_ids_dis)
                related_word_ids_dis = to_var(related_word_ids_dis)
                related_char_ids_dis = to_var(related_char_ids_dis)

            loss = seq_trainer.tagger_step(target_word_ids_dis, target_char_ids_dis, target_char_len_dis,
                                           target_seq_len_dis, target_reference_dis, related_word_ids_dis,
                                           related_char_ids_dis, related_char_len_dis, related_seq_len_dis,
                                           related_reference_dis)
        if num_batches % 100 == 0:
            print('adversarial epoch %i , current discriminator loss: %f,  current mapping loss: %f\r' %
                  (epoch, dis_loss.data[0], map_loss.data[0]))

    dev_preds = []
    for dev_batch in dev_batches:
        dev_input = create_input(dev_batch, is_cuda=params['gpu'])
        dev_word_ids = dev_input['words']
        dev_char_ids = dev_input['chars']
        dev_char_len = dev_input['char_len']
        dev_seq_len = dev_input['seq_len']
        dev_reference = dev_input['tags']

        if torch.cuda.is_available():
            dev_word_ids = to_var(dev_word_ids)
            dev_char_ids = to_var(dev_char_ids)

        dev_pred_seq = seq_trainer.tagging_dev_step(dev_word_ids, dev_char_ids, dev_char_len, dev_seq_len, dev_reference)
        dev_preds += dev_pred_seq

    dev_f1, dev_acc, dev_predicted_bio = evaluate_ner(
        params, dev_preds,
        list(itertools.chain.from_iterable(dev_batches)),
        mappings['id_to_tag'],
        mappings['id_to_word'],
        params['eval_script']
    )
    if dev_f1 > best_dev:
        best_dev = dev_f1
        current_patience = 0
        print('new best score on dev: %.4f and on test: %.4f and best on global test: %.4f  (adversarial)'
              % (best_dev, best_test, best_test_global))
        print('saving the current model to disk...')

        state = {
            'epoch': epoch + 1,
            'parameters': params,
            'mappings': mappings,
            'state_dict': module_list.state_dict(),
            'best_prec1': best_dev,
        }
        torch.save(state, os.path.join(model_dir, 'best_model.pth.tar'))
        with open(os.path.join(model_dir, 'best_dev.ner.bio'), 'w') as f:
            f.write(dev_predicted_bio)
    else:
        current_patience += 1

    print('{} epoch: {} batch: {} F1: {:.4f} Acc: {:.4f}  Current best dev: {:.4f}\n'.format(
        "dev", epoch, num_batches, dev_f1, dev_acc, best_dev))

    time_epoch_end = time.time()  # epoch end time
    print('epoch training time: %f seconds' % round(
        (time_epoch_end - time_epoch_start), 2))
