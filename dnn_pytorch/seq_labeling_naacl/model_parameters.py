import os
import json
from collections import OrderedDict


def check_params(args):
    # Parse parameters
    parameters = OrderedDict()
    parameters['signal'] = args.signal
    parameters['tag_scheme'] = args.tag_scheme
    parameters['lower'] = args.lower == 1
    parameters['zeros'] = args.zeros == 1
    parameters['char_dim'] = args.char_dim
    parameters['char_lstm_dim'] = args.char_lstm_dim
    parameters['char_cnn_dim'] = args.char_cnn_dim
    parameters['word_dim'] = args.word_dim
    parameters['word_lstm_dim'] = args.word_lstm_dim
    parameters['all_emb'] = args.all_emb == 1
    parameters['feat_path'] = args.feat_path
    parameters['crf'] = args.crf == 1
    parameters['dropout'] = args.dropout
    parameters['lr_method'] = args.lr_method
    parameters['num_epochs'] = args.num_epochs
    parameters['batch_size'] = args.batch_size
    parameters['gpu'] = args.gpu == 1
    parameters['char_conv'] = args.char_conv
    parameters['optimizer'] = args.optimizer

    # Check parameters validity
    assert os.path.isfile(args.train)
    assert os.path.isfile(args.dev)
    assert os.path.isfile(args.test)
    assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
    assert 0. <= parameters['dropout'] < 1.0
    assert parameters['tag_scheme'] in ['iob', 'iobes', 'classification']

    eval_path = os.path.join(os.path.dirname(__file__), "./evaluation")
    eval_script = os.path.join(
        eval_path, 'conlleval'
    )
    assert os.path.exists(eval_script)

    # generate model name
    model_dir = args.model_dp
    model_name = []
    for k, v in parameters.items():
        if not v:
            continue
        if k == 'pre_emb':
            v = os.path.basename(v)
        model_name.append('='.join((k, str(v))))
    model_dir = os.path.join(model_dir, ','.join(model_name[:-1]))
    os.makedirs(model_dir, exist_ok=True)

    parameters['pre_emb'] = args.pre_emb
    parameters['bi_pre_emb'] = args.bi_pre_emb
    parameters['max_word_length'] = 25
    parameters['filter_withs'] = [2, 3, 4]
    parameters['char_cnn_word_dim'] = parameters['word_dim'] + parameters['char_conv'] * len(parameters['filter_withs'])
    parameters['shared_lstm_hidden_dim'] = parameters['word_lstm_dim']

    assert not parameters['all_emb'] or parameters['pre_emb']
    assert not parameters['pre_emb'] or parameters['word_dim'] > 0
    assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

    parameters['model_dir'] = model_dir
    parameters['eval_path'] = eval_path
    parameters['eval_script'] = eval_script

    return parameters


def write_json_param_file(file):
    param = OrderedDict()
    train_param = {'default': "", 'type': "str", 'help': "train set location"}
    bi_train_param = {'default': "", 'type': "str", 'help': "bi train set location"}
    dev_param = {'default': "", 'type': "str", 'help': "dev set location"}
    test_param = {'default': "", 'type': "str", 'help': "test set location"}
    model_param = {'default': "", 'type': "str", 'help': "saved model location"}
    tag_scheme_param = {'default': "iob", 'type': "str", 'help': "tagging scheme: iob or iobes"}
    lower_param = {'default': "0", 'type': "int", 'help': "lowercase words (this will not affect character inputs)"}
    zeros_param = {'default': "0", 'type': "int", 'help': "replace digits with 0"}
    char_dim_param = {'default': "25", 'type': "int", 'help': "char embedding dimension"}
    char_lstm_dim_param = {'default': "25", 'type': "int", 'help': "char lstm hidden layer size"}
    char_cnn_param = {'default': "1", 'type': "int", 'help': "use cnn to generate character embeddings (0 disable)"}
    char_conv_param = {'default': "25", 'type': "int", 'help': "filter number"}
    word_dim_param = {'default': "100", 'type': "int", 'help': "token embedding dimension"}
    word_lstm_dim_param = {'default': "100", 'type': "int", 'help': "token lstm hidden layer size"}
    pre_emb_param = {'default': "", 'type': "str", 'help': "location of pretrained embeddings"}
    bi_pre_emb_param = {'default': "", 'type': "str", 'help': "location of pretrained related language embeddings"}
    all_emb_param = {'default': "1", 'type': "int", 'help': "load all embeddings"}
    feat_param = {'default': "0", 'type': "int", 'help': "use external features (0 to disable)"}
    crf_param = {'default': "1", 'type': "int", 'help': "use crf (0 to disable)"}
    dropout_param = {'default': "0.5", 'type': "float", 'help': "dropout on the input"}
    learning_rate_param = {'default': "0.005", 'type': "float", 'help': "learning rate for bi-lstm-crf"}
    optimizer_param = {'default': "sgd", 'type': "str", 'help': "learning method (sgd, adadelta, adam)"}
    epoch_param = {'default': "100", 'type': "int", 'help': "number of epoches for name tagger training"}
    batch_param = {'default': "20", 'type': "int", 'help': "batch size for name tagger training"}
    gpu_param = {'default': "1", 'type': "int", 'help': "use gpu. 0 to disable"}
    cuda_param = {'default': "1", 'type': "int", 'help': "cuda index"}
    model_save_signal_param = {'default': "", 'type': "str", 'help': "signal for saved model file"}

    target_lang_param = {'default': "en", 'type': "str", 'help': "target language"}
    related_lang_param = {'default': "es", 'type': "str", 'help': "related language"}
    max_vocab_param = {'default': "100", 'type': "int", 'help': "maximum vocabulary size"}
    map_id_init_param = {'default': "True", 'type': "bool", 'help': "initialize the mapping as an indentity matrix"}
    map_beta_param = {'default': "0.001", 'type': "float", 'help': "beta for orthogonalization"}
    dis_layers_param = {'default': "2", 'type': "int", 'help': "discriminator layers"}
    dis_hidden_dim_param = {'default': "2048", 'type': "int", 'help': "discriminator hidden layer dimension"}
    dis_dropout_param = {'default': "0.", 'type': "float", 'help': "dropout for discriminator layer"}
    dis_input_dropout_param = {'default': "0.1", 'type': "float", 'help': "dropout for input of discriminator layer"}
    dis_steps_param = {'default': "5", 'type': "int", 'help': "discriminator steps"}
    dis_lambda_param = {'default': "1", 'type': "float", 'help': "discriminator loss feedback coefficient"}
    dis_most_frequent_param = {'default': "30000", 'type': "int", 'help': "select embeddings of the k most frequent words for discrimination"}
    dis_smooth_param = {'default': "0.1", 'type': "float", 'help': "discriminator smooth prediction"}
    dis_clip_param = {'default': "0.", 'type': "float", 'help': "clip discriminator weights (0 to disable)"}
    adversarial_param = {'default': "True", 'type': "bool", 'help': "use adversarial training"}
    adv_epoch_param = {'default': "5", 'type': "int", 'help': "epoches adversarial training"}
    adv_epoch_iter_param = {'default': "10000", 'type': "int", 'help': "iteration per epoche for adversarial training"}
    adv_batch_param = {'default': "32", 'type': "int", 'help': "batch size adversarial training"}
    map_optimizer_param = {'default': "sgd", 'type': "str", 'help': "mapping optimizer"}
    map_learning_rate_param = {'default': "0.1", 'type': "float", 'help': "mapping learning rate"}
    dis_optimizer_param = {'default': "sgd", 'type': "str", 'help': "discriminator optimizer"}
    dis_learning_rate_param = {'default': "0.1", 'type': "float", 'help': "discriminator learning rate"}
    lr_decay_param = {'default': "0.98", 'type': "float", 'help': "learning rate decay (sgd only)"}
    mini_lr_param = {'default': "1e-6", 'type': "float", 'help': "minimum learning rate (sgd only)"}
    lr_shrink_param = {'default': "0.5", 'type': "float", 'help': "shrink the learning rate if validation metric decrease"}
    n_refinement_param = {'default': "5", 'type': "int", 'help': "number of refinement iterations"}
    dico_eval_param = {'default': "default", 'type': "str", 'help': "path to evaluation dictionary"}
    dico_method_param = {'default': "csls_knn_10", 'type': "str", 'help': "method used for dictionary generation"}
    dico_build_param = {'default': "S2T", 'type': "str", 'help': "S2T,T2S,S2T|T2S,S2T&T2S"}
    dico_threshold_param = {'default': "0.", 'type': "float", 'help': "threshold for dictionary generation"}
    dico_max_rank_param = {'default': "15000", 'type': "int", 'help': "maximum dictionary words rank"}
    dico_min_size_param = {'default': "0", 'type': "int", 'help': "minimum generated dictionary size"}
    dico_max_size_param = {'default': "0", 'type': "int", 'help': "maximum generated dictionary size"}
    target_emb_param = {'default': "", 'type': "str", 'help': "reload target language embeddings"}
    related_emb_param = {'default': "", 'type': "str", 'help': "reload related language embeddings"}
    normalize_embedding_param = {'default': "", 'type': "", 'help': "normalize embeddings before training"}

    param['train'] = train_param
    param['bi_train'] = bi_train_param
    param['dev'] = dev_param
    param['test'] = test_param
    param['model_dp'] = model_param
    param['tag_scheme'] = tag_scheme_param
    param['lower'] = lower_param
    param['zeros'] = zeros_param
    param['char_dim'] = char_dim_param
    param['char_lstm_dim'] = char_lstm_dim_param
    param['char_cnn'] = char_cnn_param
    param['char_conv'] = char_conv_param
    param['word_dim'] = word_dim_param
    param['word_lstm_dim'] = word_lstm_dim_param
    param['pre_emb'] = pre_emb_param
    param['bi_pre_emb'] = bi_pre_emb_param
    param['all_emb'] = all_emb_param
    param['feat'] = feat_param
    param['crf'] = crf_param
    param['dropout'] = dropout_param
    param['learning_rate'] = learning_rate_param
    param['optimizer'] = optimizer_param
    param['num_epoches'] = epoch_param
    param['batch_size'] = batch_param
    param['gpu'] = gpu_param
    param['cuda'] = cuda_param
    param['signal'] = model_save_signal_param

    param['target_lang'] = target_lang_param
    param['related_lang'] = related_lang_param
    param['max_vocab'] = max_vocab_param
    param['map_id_init'] = map_id_init_param
    param['map_beta'] = map_beta_param
    param['dis_layers'] = dis_layers_param
    param['dis_hid_dim'] = dis_hidden_dim_param
    param['dis_dropout'] = dis_dropout_param
    param['dis_input_dropout'] = dis_input_dropout_param
    param['dis_steps'] = dis_steps_param
    param['dis_lambda'] = dis_lambda_param
    param['dis_most_frequent'] = dis_most_frequent_param
    param['dis_smooth'] = dis_smooth_param
    param['dis_clip_weights'] = dis_clip_param
    param['adversarial'] = adversarial_param
    param['adv_epochs'] = adv_epoch_param
    param['adv_batch_size'] = adv_batch_param
    param['map_learning_rate'] = map_learning_rate_param
    param['map_optimizer'] = map_optimizer_param
    param['dis_learning_rate'] = dis_learning_rate_param
    param['dis_optimizer'] = dis_optimizer_param
    param['lr_decay'] = lr_decay_param
    param['min_lr'] = mini_lr_param
    param['lr_shrink'] = lr_shrink_param
    param['n_refinement'] = n_refinement_param
    param['dico_eval'] = dico_eval_param
    param['dico_method'] = dico_method_param
    param['dico_build'] = dico_build_param
    param['dico_threshold'] = dico_threshold_param
    param['dico_max_rank'] = dico_max_rank_param
    param['dico_min_size'] = dico_min_size_param
    param['dico_max_size'] = dico_max_size_param
    param['target_emb'] = target_emb_param
    param['related_emb'] = related_emb_param
    param['normalize_embeddings'] = normalize_embedding_param

    with open(file, 'w') as out_file:
        str_ = json.dumps(param, sort_keys=False, indent=4)
        out_file.write(str_)


def load_json_param(file):
    param = {}
    with open(file) as param_file:
        data_load = json.load(param_file)

    for key in data_load:
        value_type = data_load[key]['type']
        default_value = data_load[key]['default']
        help = data_load[key]['help']
        if value_type == "int":
            param[key] = int(default_value)
        elif value_type == 'float':
            param[key] = float(default_value)
        elif value_type == 'str':
            param[key] = str(default_value)
        elif value_type == 'bool':
            if default_value == 'False' or default_value == '0':
                param[key] = False
            else:
                param[key] = True
            # param[key] = bool(default_value)
        print(type(param[key]))
        print("key: " + key + " value: " + str(param[key]))

        #print("key: " + key + " type: " + type(param[key] + " value: " + str(param[key])))
    return param

if __name__ == '__main__':
    # write_json_param_file("/Users/lifu/Documents/RPI/Research/Code/PythonWorkspace/name_tagging-master/"
    #                 "example/params/model_params.json")
    load_json_param("/Users/lifu/Documents/RPI/Research/Code/PythonWorkspace/name_tagging-master/"
                    "example/params/model_params.json")