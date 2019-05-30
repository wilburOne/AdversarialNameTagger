import codecs
import os

import torch
import torch.utils.data as data


class BilingualAlignment(data.Dataset):
    def __init__(self, dict_file, vocab1, vocab2):
        self.dict_file = dict_file
        dict1, dict2 = self.__load_dict__(dict_file, vocab1, vocab2)
        self.dict1 = dict1
        self.dict2 = dict2
        self.ids = list(self.dict1.keys())

    def __load_dict__(self, dict_file, vocab1, vocab2):
        dict1 = {}
        dict2 = {}
        index = 0

        with codecs.open(dict_file, 'r', 'utf-8') as f:
            for line in f:
                parts = line.lower().strip().split("\t")
                word1 = parts[0][3:].strip()
                word2 = parts[1][3:].strip()

                if word1 in vocab1 and word2 in vocab2:
                    word_index1 = vocab1[word1]
                    word_index2 = vocab2[word2]
                    dict1[index] = word_index1
                    dict2[index] = word_index2
                    index += 1

        print("{} pairs of word alignment are loaded".format(len(dict1)))
        return dict1, dict2

    def __getitem__(self, index):
        """Returns one data."""
        id1 = self.dict1[index]
        id2 = self.dict2[index]

        return id1, id2

    def __len__(self):
        return len(self.ids)


def collate_fn_bilingual(data):
    """Creates mini-batch tensors from the list of tuples.
    """
    ids1, ids2 = zip(*data)
    return ids1, ids2


def get_loader_bilingual(dict_path, vocab1, vocab2, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom input dataset."""

    data_loaders = []

    data_set = BilingualAlignment(dict_path, vocab1, vocab2)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_bilingual)
    return data_loader


class Monolingual(data.Dataset):
    def __init__(self, word2vec_file, vocab_lang, context_lang, word2char_lang, head, topk):
        self.vocab_lang = vocab_lang
        self.context_lang = context_lang
        self.topk = topk
        dict1, context_dict1, char_dict1 = self.__load_dict__(word2vec_file, vocab_lang, context_lang,
                                                              word2char_lang, head, topk)
        self.dict1 = dict1
        self.context_dict1 = context_dict1
        self.char_dict1 = char_dict1
        self.ids = list(self.dict1.keys())

    def __load_dict__(self, word2vec_file, vocab_lang, context_lang, word2char_lang, head, topk):
        dict1 = {}
        context_dict = {}
        char_dict1 = {}
        words = {}
        index = 0
        count = 0
        topK = topk
        with codecs.open(word2vec_file, 'r', 'utf-8') as f:
            for line in f:
                if head is True and count == 0:
                    count = 1
                    continue
                else:
                    parts = line.split(" ")
                    word1 = parts[0].strip()
                    if word1 in vocab_lang.word2idx and word1 in context_lang:
                        word_index1 = vocab_lang.word2idx[word1]
                        dict1[index] = word_index1
                        char_index1 = word2char_lang[word_index1]
                        char_dict1[index] = char_index1

                        context_tmp_words = context_lang[word1]
                        context_words = []
                        count1 = 0
                        for tmp1 in context_tmp_words:
                            if count1 < topK and tmp1 in vocab_lang.word2idx:
                                context_words.append(vocab_lang.word2idx[tmp1])
                                count1 += 1
                        if len(context_words) > 0 and len(word1) > 0:
                            context_dict[index] = context_words
                            words[index] = word1
                            index += 1
        print("{} words are loaded ... ".format(len(context_dict)))
        return dict1, context_dict, char_dict1

    def __getitem__(self, index):
        """Returns one data pair."""
        id1 = self.dict1[index]
        context_id1 = self.context_dict1[index]
        char_id1 = self.char_dict1[index]
        return index, id1, context_id1, char_id1

    def __len__(self):
        return len(self.ids)


def collate_fn_monolingual(data):
    """Creates mini-batch tensors from the list of tuples."""

    idxs, ids, ids1_context, char_ids1 = zip(*data)
    lengths1 = [len(ids1_context_tmp) for ids1_context_tmp in ids1_context]
    ids1_context_targets = torch.zeros(len(ids1_context), max(lengths1)).float()
    for i, ids1_context_tmp in enumerate(ids1_context):
        end = lengths1[i]
        ids1_context_tmp = torch.FloatTensor(ids1_context_tmp)
        ids1_context_targets[i, :end] = ids1_context_tmp[:end]

    lengths1 = [len(char_ids1_tmp) for char_ids1_tmp in char_ids1]
    char_ids1_targets = torch.zeros(len(char_ids1), max(lengths1)).float()
    for i, char_ids1_tmp in enumerate(char_ids1):
        end = lengths1[i]
        char_ids1_tmp = torch.from_numpy(char_ids1_tmp).float()
        char_ids1_targets[i, :end] = char_ids1_tmp[:end]

    return ids, ids1_context_targets, char_ids1_targets


def get_loader_monolingual(word2vec_file, vocab_lang, context_lang, word2char_lang, head, batch_size,
                                      shuffle, num_workers, topk):
    """Returns torch.utils.data.DataLoader for custom input dataset."""

    data_set = Monolingual(word2vec_file, vocab_lang, context_lang, word2char_lang, head, topk)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn_monolingual)
    return data_loader
