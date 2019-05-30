import os
from collections import OrderedDict


def parse_args(args):
    # Parse parameters
    params = OrderedDict()
    params['train'] = args.train
    params['bi_train'] = args.bi_train
    params['dev'] = args.dev
    params['test'] = args.test
    params['model_dp'] = args.model_dp
    params['tag_scheme'] = args.tag_scheme
    params['lower'] = args.lower == 1
    params['zeros'] = args.zeros == 1
    params['char_dim'] = args.char_dim
    params['char_lstm_dim'] = args.char_lstm_dim
    params['char_cnn'] = args.char_cnn
    params['char_conv'] = args.char_conv
    params['word_dim'] = args.word_dim
    params['word_lstm_dim'] = args.word_lstm_dim
    params['pre_emb'] = args.pre_emb
    params['bi_pre_emb'] = args.bi_pre_emb
    params['all_emb'] = args.all_emb == 1
    params['feat'] = args.feat
    params['crf'] = args.crf == 1
    params['dropout'] = args.dropout
    params['tagger_learning_rate'] = args.tagger_learning_rate
    params['tagger_optimizer'] = args.tagger_optimizer
    params['dis_seq_learning_rate'] = args.dis_seq_learning_rate
    params['dis_seq_optimizer'] = args.dis_seq_optimizer
    params['mapping_seq_learning_rate'] = args.mapping_seq_learning_rate
    params['mapping_seq_optimizer'] = args.mapping_seq_optimizer
    params['num_epochs'] = args.num_epochs
    params['batch_size'] = args.batch_size
    params['gpu'] = args.gpu
    params['cuda'] = args.cuda
    params['signal'] = args.signal
    params['target_lang'] = args.target_lang
    params['related_lang'] = args.related_lang
    params['max_vocab'] = args.max_vocab
    params['map_id_init'] = args.map_id_init
    params['map_beta'] = args.map_beta
    params['dis_layers'] = args.dis_layers
    params['dis_hid_dim'] = args.dis_hid_dim
    params['dis_dropout'] = args.dis_dropout
    params['dis_input_dropout'] = args.dis_input_dropout
    params['dis_steps'] = args.dis_steps
    params['dis_lambda'] = args.dis_lambda
    params['dis_most_frequent'] = args.dis_most_frequent
    params['dis_smooth'] = args.dis_smooth
    params['dis_clip_weights'] = args.dis_clip_weights
    params['adversarial'] = args.adversarial
    params['adv_epochs'] = args.adv_epochs
    params['adv_iteration'] = args.adv_iteration
    params['adv_batch_size'] = args.adv_batch_size
    params['map_learning_rate'] = args.map_learning_rate
    params['map_optimizer'] = args.map_optimizer
    params['dis_learning_rate'] = args.dis_learning_rate
    params['dis_optimizer'] = args.dis_optimizer
    params['lr_decay'] = args.lr_decay
    params['min_lr'] = args.min_lr
    params['lr_shrink'] = args.lr_shrink
    params['n_refinement'] = args.n_refinement
    params['dico_eval'] = args.dico_eval
    params['dico_method'] = args.dico_method
    params['dico_build'] = args.dico_build
    params['dico_threshold'] = args.dico_threshold
    params['dico_max_rank'] = args.dico_max_rank
    params['dico_min_size'] = args.dico_min_size
    params['dico_max_size'] = args.dico_max_size
    params['target_emb'] = args.target_emb
    params['related_emb'] = args.related_emb
    params['normalize_embeddings'] = args.normalize_embeddings

    # Check parameters validity
    assert os.path.isfile(args.dev)
    assert os.path.isfile(args.test)
    assert params['char_dim'] > 0 or params['word_dim'] > 0
    assert 0. <= params['dropout'] < 1.0
    assert params['tag_scheme'] in ['iob', 'iobes', 'classification']
    assert not params['all_emb'] or params['pre_emb']
    assert not params['pre_emb'] or params['word_dim'] > 0
    assert not params['pre_emb'] or os.path.isfile(params['pre_emb'])

    model_names = OrderedDict()
    model_names = {'signal': args.signal, 'tag_scheme': args.tag_scheme, 'lower': args.lower == 1,
                   'zeros': args.zeros == 1, 'char_dim': args.char_dim, 'char_lstm_dim': args.char_lstm_dim,
                   'char_conv': args.char_conv, 'word_dim': args.word_dim, 'word_lstm_dim': args.word_lstm_dim,
                   'pre_emb': args.pre_emb, 'all_emb': args.all_emb == 1, 'crf': args.crf == 1,
                   'dropout': args.dropout, 'tagger_learning_rate': args.tagger_learning_rate, 'num_epochs': args.num_epochs,
                   'batch_size': args.batch_size, 'tagger_optimizer': args.tagger_optimizer}

    return params, model_names
