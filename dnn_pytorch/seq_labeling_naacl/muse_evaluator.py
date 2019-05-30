# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# from logging import getLogger
# from copy import deepcopy
# import numpy as np
# from torch.autograd import Variable
# from torch import Tensor as torch_tensor
# import torch
#
# from . import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
# from . import get_word_translation_accuracy
# from . import load_europarl_data, get_sent_translation_accuracy
# from src.utils import get_idf
# from ..utils import get_nn_avg_dist
from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch import Tensor as torch_tensor
import torch

from wordsim import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from word_translation import get_word_translation_accuracy
from sent_translation import load_europarl_data, get_sent_translation_accuracy
from muse_utils import get_idf
from muse_utils import get_nn_avg_dist


def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params['dico_max_rank'] > 0 and not params['dico_method'].startswith('invsm_beta_'):
        n_src = params['dico_max_rank']

    # nearest neighbors
    if params['dico_method'] == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif params['dico_method'].startswith('invsm_beta_'):

        beta = float(params['dico_method'][len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params['dico_method'].startswith('csls_knn_'):

        knn = params['dico_method'][len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params['dico_max_rank'] > 0:
        selected = all_pairs.max(1)[0] <= params['dico_max_rank']
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params['dico_max_size'] > 0:
        all_scores = all_scores[:params['dico_max_size']]
        all_pairs = all_pairs[:params['dico_max_size']]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params['dico_min_size'] > 0:
        diff[:params['dico_min_size']] = 1e9

    # confidence threshold
    if params['dico_threshold'] > 0:
        mask = diff > params['dico_threshold']
        print("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs


def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    print("Building the train dictionary ...")
    s2t = 'S2T' in params['dico_build']
    t2s = 'T2S' in params['dico_build']
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params['dico_build'] == 'S2T':
        dico = s2t_candidates
    elif params['dico_build'] == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
        if params['dico_build'] == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params['dico_build'] == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                return None
        dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    print('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda()


def monolingual_wordsim(src_dico, tgt_dico, projected_src_emb, tgt_emb):
    """
    Evaluation on monolingual word similarity.
    """
    metrics = {}
    print("monolingual word sim eval ... ")
    src_ws_scores = get_wordsim_scores(
        src_dico.lang, src_dico.word2id,
        projected_src_emb.data.cpu().numpy()
    )
    tgt_ws_scores = get_wordsim_scores(
        tgt_dico.lang, tgt_dico.word2id,
        tgt_emb.weight.data.cpu().numpy()
    )
    if src_ws_scores is not None:
        src_ws_monolingual_scores = np.mean(list(src_ws_scores.values()))
        print("Monolingual source word similarity score average: %.5f" % src_ws_monolingual_scores)
        metrics['src_ws_monolingual_scores'] = src_ws_monolingual_scores
        metrics.update({'src_' + k: v for k, v in src_ws_scores.items()})
    if tgt_ws_scores is not None:
        tgt_ws_monolingual_scores = np.mean(list(tgt_ws_scores.values()))
        print("Monolingual target word similarity score average: %.5f" % tgt_ws_monolingual_scores)
        metrics['tgt_ws_monolingual_scores'] = tgt_ws_monolingual_scores
        metrics.update({'tgt_' + k: v for k, v in tgt_ws_scores.items()})
    if src_ws_scores is not None and tgt_ws_scores is not None:
        ws_monolingual_scores = (src_ws_monolingual_scores + tgt_ws_monolingual_scores) / 2
        print("Monolingual word similarity score average: %.5f" % ws_monolingual_scores)
        metrics['ws_monolingual_scores'] = ws_monolingual_scores

    return metrics


def monolingual_wordanalogy(src_dico, tgt_dico, projected_src_emb, tgt_emb):
    """
    Evaluation on monolingual word analogy.
    """
    metrics = {}
    print("monolingual word analogy eval ... ")
    src_analogy_scores = get_wordanalogy_scores(
        src_dico.lang, src_dico.word2id,
        projected_src_emb.data.cpu().numpy()
    )
    tgt_analogy_scores = get_wordanalogy_scores(
        tgt_dico.lang, tgt_dico.word2id,
        tgt_emb.weight.data.cpu().numpy()
    )
    if src_analogy_scores is not None:
        src_analogy_monolingual_scores = np.mean(list(src_analogy_scores.values()))
        print("Monolingual source word analogy score average: %.5f" % src_analogy_monolingual_scores)
        metrics['src_analogy_monolingual_scores'] = src_analogy_monolingual_scores
        metrics.update({'src_' + k: v for k, v in src_analogy_scores.items()})
    if tgt_analogy_scores is not None:
        tgt_analogy_monolingual_scores = np.mean(list(tgt_analogy_scores.values()))
        print("Monolingual target word analogy score average: %.5f" % tgt_analogy_monolingual_scores)
        metrics['tgt_analogy_monolingual_scores'] = tgt_analogy_monolingual_scores
        metrics.update({'tgt_' + k: v for k, v in tgt_analogy_scores.items()})

    return metrics


def crosslingual_wordsim(src_dico, tgt_dico, projected_src_emb, tgt_emb):
    """
    Evaluation on cross-lingual word similarity.
    """
    metrics = {}
    print("cross lingual word sim eval ... ")
    src_emb = projected_src_emb.data.cpu().numpy()
    tgt_emb = tgt_emb.weight.data.cpu().numpy()
    # cross-lingual wordsim evaluation
    src_tgt_ws_scores = get_crosslingual_wordsim_scores(
        src_dico.lang, src_dico.word2id, src_emb,
        tgt_dico.lang, tgt_dico.word2id, tgt_emb,
    )
    if src_tgt_ws_scores is None:
        return
    ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
    print("Cross-lingual word similarity score average: %.5f" % ws_crosslingual_scores)
    metrics['ws_crosslingual_scores'] = ws_crosslingual_scores
    metrics.update({'src_tgt_' + k: v for k, v in src_tgt_ws_scores.items()})

    return metrics


def word_translation(src_dico, tgt_dico, projected_src_emb, tgt_emb, dico_eval):
    """
    Evaluation on word translation.
    """
    metrics = {}
    print("word translation eval ... ")
    # mapped word embeddings
    src_emb = projected_src_emb.data
    tgt_emb = tgt_emb.weight.data

    for method in ['nn', 'csls_knn_10']:
        results = get_word_translation_accuracy(
            src_dico.lang, src_dico.word2id, src_emb,
            tgt_dico.lang, tgt_dico.word2id, tgt_emb,
            method=method,
            dico_eval=dico_eval
        )
        metrics.update([('%s-%s' % (k, method), v) for k, v in results])

    return metrics


def sent_translation(src_dico, tgt_dico, projected_src_emb, tgt_emb):
    """
    Evaluation on sentence translation.
    Only available on Europarl, for en - {de, es, fr, it} language pairs.
    """
    metrics = {}
    print("sent translation eval ... ")
    lg1 = src_dico.lang
    lg2 = tgt_dico.lang

    # parameters
    n_keys = 200000
    n_queries = 2000
    n_idf = 300000

    # load europarl data
    if not hasattr('europarl_data'):
        europarl_data = load_europarl_data(
            lg1, lg2, n_max=(n_keys + 2 * n_idf)
        )

    # if no Europarl data for this language pair
    if not europarl_data:
        return

    # mapped word embeddings
    src_emb = projected_src_emb.data
    tgt_emb = tgt_emb.weight.data

    # get idf weights
    idf = get_idf(europarl_data, lg1, lg2, n_idf=n_idf)

    for method in ['nn', 'csls_knn_10']:
        # source <- target sentence translation
        results = get_sent_translation_accuracy(
            europarl_data,
            src_dico.lang, src_dico.word2id, src_emb,
            tgt_dico.lang, tgt_dico.word2id, tgt_emb,
            n_keys=n_keys, n_queries=n_queries,
            method=method, idf=idf
        )
        metrics.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

        # target <- source sentence translation
        results = get_sent_translation_accuracy(
            europarl_data,
            tgt_dico.lang, tgt_dico.word2id, tgt_emb,
            src_dico.lang, src_dico.word2id, src_emb,
            n_keys=n_keys, n_queries=n_queries,
            method=method, idf=idf
        )
        metrics.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])

    return metrics


def dist_mean_cosine(src_dico, tgt_dico, projected_src_emb, tgt_emb):
    """
    Mean-cosine model selection criterion.
    """
    metrics = {}
    print("dist mean cosine eval ... ")
    # get normalized embeddings
    src_emb = projected_src_emb.data
    tgt_emb = tgt_emb.weight.data
    src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
    tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

    # build dictionary
    for dico_method in ['nn', 'csls_knn_10']:
        dico_build = 'S2T'
        dico_max_size = 10000
        # temp params / dictionary generation
        _params = {}
        _params['dico_method'] = dico_method
        _params['dico_build'] = dico_build
        _params['dico_threshold'] = 0
        _params['dico_max_rank'] = 10000
        _params['dico_min_size'] = 0
        _params['dico_max_size'] = dico_max_size
        s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
        t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
        dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
        # mean cosine
        if dico is None:
            mean_cosine = -1e9
        else:
            mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
        mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
        print("Mean cosine (%s method, %s build, %i max size): %.5f"
              % (dico_method, _params['dico_build'], dico_max_size, mean_cosine))
        key = 'mean_cosine-' + dico_method + '-' + _params['dico_build'] + '-' + str(dico_max_size)
        metrics[key] = mean_cosine

    return metrics


def all_eval(src_dico, tgt_dico, projected_src_emb, tgt_emb, dico_eval, VALIDATION_METRIC):
    """
    Run all evaluations.
    """
    valid_metric = 0
    all_metrics = {}

    # metrics1 = monolingual_wordsim(src_dico, tgt_dico, projected_src_emb, tgt_emb)
    # all_metrics.update(metrics1)
    #
    # print("all metrics ...  ", str(all_metrics))
    #
    # metrics2 = crosslingual_wordsim(src_dico, tgt_dico, projected_src_emb, tgt_emb)
    # # all_metrics = all_metrics.update(metrics2)
    #
    # print("all metrics ...  ", str(all_metrics))
    #
    # metrics3 = word_translation(src_dico, tgt_dico, projected_src_emb, tgt_emb, dico_eval)
    # all_metrics.update(metrics3)
    #
    # print("all metrics ...  ", str(all_metrics))

    # metrics4 = sent_translation(src_dico, tgt_dico, projected_src_emb, tgt_emb)
    metrics5 = dist_mean_cosine(src_dico, tgt_dico, projected_src_emb, tgt_emb)
    all_metrics.update(metrics5)

    print("all metrics ...  ", str(all_metrics))
    valid_metric = all_metrics[VALIDATION_METRIC]
    return valid_metric


def eval_dis(discriminator, mapping, src_emb, tgt_emb):
    """
    Evaluate discriminator predictions and accuracy.
    """
    print("discriminator eval ... ")
    bs = 128
    src_preds = []
    tgt_preds = []

    discriminator.eval()

    for i in range(0, src_emb.num_embeddings, bs):
        emb = Variable(src_emb.weight[i:i + bs].data, volatile=True)
        print("emb", emb)
        preds = discriminator(mapping(emb))
        src_preds.extend(preds.data.cpu().tolist())

    for i in range(0, tgt_emb.num_embeddings, bs):
        emb = Variable(tgt_emb.weight[i:i + bs].data, volatile=True)
        preds = discriminator(emb)
        tgt_preds.extend(preds.data.cpu().tolist())

    src_pred = np.mean(src_preds)
    tgt_pred = np.mean(tgt_preds)
    print("Discriminator source / target predictions: %.5f / %.5f"% (src_pred, tgt_pred))

    src_accu = np.mean([x >= 0.5 for x in src_preds])
    tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
    dis_accu = ((src_accu * src_emb.num_embeddings + tgt_accu * tgt_emb.num_embeddings) /
                (src_emb.num_embeddings + tgt_emb.num_embeddings))
    print("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
          % (src_accu, tgt_accu, dis_accu))

