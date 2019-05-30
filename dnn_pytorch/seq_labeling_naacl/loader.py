import os
import gzip
import lzma
import re
import logging
import numpy as np

import torch
import torch.nn as nn

from preprocessing import create_dico, create_mapping


def load_sentences(path, to_sort=False):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in open(path, 'r'):
        # if len(sentences) > 100:
        #     break

        line = line.strip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            sentence.append(word)

    if len(sentence) > 0:
        sentences.append(sentence)

    # sort sentences by length
    if to_sort:
        sentences = sorted(sentences, key=lambda x: len(x))

    return sentences


def load_pretrained_emb_weights(id_to_word, pre_emb, word_dim):
    if not pre_emb:
        return

    new_weights = np.random.normal(scale=0.6, size=(len(id_to_word), word_dim))

    # Initialize with pretrained embeddings
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
            pretrained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
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
            new_weights[i] = pretrained[word]
            c_found += 1
        elif word.lower() in pretrained:
            new_weights[i] = pretrained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pretrained:
            new_weights[i] = pretrained[re.sub('\d', '0', word.lower())]
            c_zeros += 1

    print('Loaded %i pretrained embeddings.' % len(pretrained))
    print('%i / %i (%.4f%%) words have been initialized with pretrained embeddings.' % (
                  c_found + c_lower + c_zeros, len(id_to_word), 100. * (c_found + c_lower + c_zeros) / len(id_to_word)))
    print('%i found directly, %i after lowercasing, %i after lowercasing + zero.' % (c_found, c_lower, c_zeros))

    return new_weights


def load_pretrained(word_emb, id_to_word, pre_emb, **kwargs):
    if not pre_emb:
        return

    word_dim = word_emb.weight.size(1)

    # Initialize with pretrained embeddings
    new_weights = word_emb.weight.data
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
    word_emb.weight = nn.Parameter(new_weights)

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
    return word_emb


def load_embedding(pre_emb):
    if os.path.basename(pre_emb).endswith('.xz'):
        return lzma.open(pre_emb)
    if os.path.basename(pre_emb).endswith('.gz'):
        return gzip.open(pre_emb, 'rb')
    else:
        return open(pre_emb, 'r', errors='replace')


def augment_with_pretrained(all_sentences, ext_emb_path, lower=False):
    """
    Augment the dictionary with words that have a pretrained embedding.
    """
    print(
        'Augmenting words by pretrained embeddings from %s...' % ext_emb_path
    )
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = []
    if len(ext_emb_path) > 0:
        for line in load_embedding(ext_emb_path):
            if not line.strip():
                continue
            if type(line) == bytes:
                try:
                    pretrained.append(str(line, 'utf-8').rstrip().split()[0].strip())
                except UnicodeDecodeError:
                    continue
            else:
                pretrained.append(line.rstrip().split()[0].strip())

    pretrained = set(pretrained)

    # preprocess sentence with zero or lower flag
    for i, s in enumerate(all_sentences):
        for j, w in enumerate(s):
            if lower:
                w[0] = w[0].lower()
            all_sentences[i][j][0] = w[0]

    words = [[x[0] for x in s] for s in all_sentences]
    dico = create_dico(words)

    # dico = create_dico(all_sentences)

    # We add every word in the pretrained file
    for word in pretrained:
        if word not in dico:
            dico[word] = 0

    word_to_id, id_to_word = create_mapping(dico)

    mappings = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word
    }

    return mappings


def augment_with_pretrained_bi(all_sentences, ext_emb_path, lower=False):
    """
    Augment the dictionary with words that have a pretrained embedding.
    """
    print(
        'Augmenting words by pretrained embeddings from %s...' % ext_emb_path
    )
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = []
    if len(ext_emb_path) > 0:
        for line in load_embedding(ext_emb_path):
            if not line.strip():
                continue
            if type(line) == bytes:
                try:
                    pretrained.append(str(line, 'utf-8').rstrip().split()[0].strip())
                except UnicodeDecodeError:
                    continue
            else:
                pretrained.append(line.rstrip().split()[0].strip())

    pretrained = set(pretrained)

    # preprocess sentence with zero or lower flag
    for i, s in enumerate(all_sentences):
        for j, w in enumerate(s):
            if lower:
                w[0] = w[0].lower()
            all_sentences[i][j][0] = w[0]

    words = [[x[0] for x in s] for s in all_sentences]
    dico = create_dico(words)

    # dico = create_dico(all_sentences)

    # We add every word in the pretrained file
    for word in pretrained:
        if word not in dico:
            dico[word] = 0

    word_to_id, id_to_word = create_mapping(dico)

    mappings = {
        'bi_word_to_id': word_to_id,
        'bi_id_to_word': id_to_word
    }

    return mappings


def load_features(feat_path):
    return []

