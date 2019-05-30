import re
import torch
from torch.autograd import Variable
import numpy as np
from collections import defaultdict


def create_input(data, is_train=True, is_cuda=True):
    """
    Take sentence data and return input variables for
    the pytorch training or evaluation function.
    """
    input = defaultdict(list)

    for d in data:
        words = d['words']

        chars = d['chars']

        input['words'].append(words)
        input['chars'].append(chars)

        if d['feats']:
            input['feats'].append(d['feats'])

        if is_train:
            tags = d['tags']
            input['tags'].append(tags)

    updated_input = {}
    for k, v in input.items():
        if k == 'chars':
            padded_chars, char_len = pad_chars(v)
            updated_input[k] = padded_chars
            updated_input['char_len'] = char_len
        elif k in ['words', 'feats', 'tags']:
            padded_words, seq_len = pad_word(v)
            updated_input[k] = padded_words
            updated_input['seq_len'] = seq_len
    input = updated_input

    # convert input and labels to Variable
    for k, v in input.items():
        if is_cuda:
            input[k] = Variable(
                torch.from_numpy(v).type(torch.cuda.LongTensor)
            )
        else:
            input[k] = Variable(
                torch.from_numpy(v).type(torch.LongTensor)
            )

    return input


def pad_word(input):
    seq_len = np.zeros(len(input), dtype=int)
    for i, s in enumerate(input):
        seq_len[i] = len(s)

    padded_input = np.zeros((len(input), max(seq_len)))
    for i, s in enumerate(input):
        padded_input[i][:seq_len[i]] = np.array(s)

    return padded_input, seq_len


def pad_chars(input):
    max_char_len = 30
    seq_len = np.zeros(len(input), dtype=int)
    for i, s in enumerate(input):
        seq_len[i] = len(s)

    char_len = np.zeros((len(input), max(seq_len)))
    for i, s in enumerate(input):
        for j, w in enumerate(s):
            char_len[i][j] = len(w) if len(w) < max_char_len else max_char_len

    padded_input = np.zeros((len(input), max(seq_len), max_char_len))

    for i, s in enumerate(input):
        for j, w in enumerate(s):
            for k, c in enumerate(w):
                padded_input[i][j][k] = c

    return padded_input, char_len


def prepare_dataset(sentences, mappings,
                    zeros=False, lower=False, is_train=True):
    # retrive mappinsg
    word_to_id = mappings['word_to_id']
    char_to_id = mappings['char_to_id']
    tag_to_id = mappings['tag_to_id']
    feat_to_id_list = mappings['feat_to_id_list']

    # preapre dataset
    data = []
    for i, s in enumerate(sentences):
        data.append(
            prepare_sentence(
                s, word_to_id, char_to_id, tag_to_id, feat_to_id_list,
                zeros=zeros, lower=lower, is_train=is_train
            )
        )

    return data


def prepare_dataset_bi(sentences, mappings,
                    zeros=False, lower=False, is_train=True):
    # retrive mappinsg
    word_to_id = mappings['bi_word_to_id']
    char_to_id = mappings['char_to_id']
    tag_to_id = mappings['tag_to_id']
    feat_to_id_list = mappings['feat_to_id_list']

    # preapre dataset
    data = []
    for i, s in enumerate(sentences):
        data.append(
            prepare_sentence(
                s, word_to_id, char_to_id, tag_to_id, feat_to_id_list,
                zeros=zeros, lower=lower, is_train=is_train
            )
        )

    return data


def prepare_mapping_bi(sentences, bi_sentences, features, zeros=False, lower=False, **kwargs):
    """
    prepare word2id, char2id, tag2id, feat2id mappings

    :param sentences:
    :param features:
    :return:
    """
    # preprocess sentence with zero or lower flag
    for i, s in enumerate(sentences):
        for j, w in enumerate(s):
            if zeros:
                w[0] = zero_digits(w[0])
            if lower:
                w[0] = w[0].lower()
            sentences[i][j][0] = w[0]

    words = [[x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))

    for i, s in enumerate(bi_sentences):
        for j, w in enumerate(s):
            if zeros:
                w[0] = zero_digits(w[0])
            if lower:
                w[0] = w[0].lower()
            bi_sentences[i][j][0] = w[0]

    bi_words = [[x[0] for x in s] for s in bi_sentences]
    bi_dico = create_dico(bi_words)
    bi_dico['<UNK>'] = 10000000
    bi_word_to_id, bi_id_to_word = create_mapping(bi_dico)
    print("Found %i unique words (%i in total)" % (
        len(bi_dico), sum(len(x) for x in bi_words)
    ))

    all_sentence = sentences + bi_sentences

    chars = ["".join([w[0] for w in s]) for s in all_sentence]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))

    tags = [[word[-1] for word in s] for s in all_sentence]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    start_index = len(tag_to_id)
    stop_index = start_index + 1
    tag_to_id[START_TAG] = start_index
    id_to_tag[start_index] = START_TAG
    tag_to_id[STOP_TAG] = stop_index
    id_to_tag[stop_index] = STOP_TAG

    feat_to_id_list = []
    id_to_feat_list = []

    mappings = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'bi_word_to_id': bi_word_to_id,
        'bi_id_to_word': bi_id_to_word,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'tag_to_id': tag_to_id,
        'id_to_tag': id_to_tag,
        'feat_to_id_list': feat_to_id_list,
        'id_to_feat_list': id_to_feat_list,

    }
    return mappings


def prepare_mapping(sentences, features, zeros=False, lower=False, **kwargs):
    """
    prepare word2id, char2id, tag2id, feat2id mappings

    :param sentences:
    :param features:
    :return:
    """
    # preprocess sentence with zero or lower flag
    for i, s in enumerate(sentences):
        for j, w in enumerate(s):
            if zeros:
                w[0] = zero_digits(w[0])
            if lower:
                w[0] = w[0].lower()
            sentences[i][j][0] = w[0]

    words = [[x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))

    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))

    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    start_index = len(tag_to_id)
    stop_index = start_index + 1
    tag_to_id[START_TAG] = start_index
    id_to_tag[start_index] = START_TAG
    tag_to_id[STOP_TAG] = stop_index
    id_to_tag[stop_index] = STOP_TAG

    feat_to_id_list = []
    id_to_feat_list = []

    mappings = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'char_to_id': char_to_id,
        'id_to_char': id_to_char,
        'tag_to_id': tag_to_id,
        'id_to_tag': id_to_tag,
        'feat_to_id_list': feat_to_id_list,
        'id_to_feat_list': id_to_feat_list,

    }
    return mappings


def prepare_sentence(sentence, word_to_id, char_to_id, tag_to_id,
                     feat_to_id_list=None,
                     zeros=False, lower=False, is_train=True):
    """
        Prepare a sentence for evaluation.
    """
    # preprocess sentence with zero or lower flag
    for i, w in enumerate(sentence):
        if zeros:
            w[0] = zero_digits(w[0])
        if lower:
            w[0] = w[0].lower()
        sentence[i][0] = w[0]

    # prepare words
    max_sent_len = 150
    if is_train:
        sentence = sentence[:max_sent_len]

    # words = [word_to_id[w[0] if w[0] in word_to_id else '<UNK>']
    #          for w in sentence]
    words = [word_to_id[w[0] if w[0] in word_to_id else 'unk']
             for w in sentence]

    # prepare chars
    # set max char len for char embedding layer to save memory
    max_char_len = 30
    chars = [
        [char_to_id[c if c in char_to_id else '<UNK>']
         for c in w[0][:max_char_len]]
        for w in sentence
        ]
    tags = []
    if is_train:
        for w in sentence:
            if w[-1] in tag_to_id:
                tags.append(tag_to_id[w[-1]])
            else:
                tags.append(0)

    # features
    sent_feats = []
    if feat_to_id_list:
        end_column = len(sentence[0])
        for token in sentence:
            s_feats = []
            for j, feat in enumerate(token[feat_column:end_column]):
                if feat not in feat_to_id_list[j]:
                    s_feats.append(feat_to_id_list[j]['<UNK>'])
                else:
                    s_feats.append(feat_to_id_list[j][feat])
            if s_feats:
                sent_feats.append(s_feats)

    return {
        'words': words,
        'chars': chars,
        'tags': tags,
        'feats': sent_feats,
    }


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)