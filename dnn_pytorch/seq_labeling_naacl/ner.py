import sys
import itertools
import collections
import argparse
import shutil

from loader import load_sentences


#
# generate external features
#
def generate_features(sentences, parameters):
    feats = []
    if parameters['pos']:
        feats.append(postag_feature(sentences, parameters))
    if parameters['cluster']:
        feats.append(clustering_feature(sentences, parameters))

    feats = merge_features(feats)

    return feats


#
# clustering features
#
def clustering_feature(sentences, parameters):
    def f(x): return x.lower()
    cluster_path_file = parameters['cluster']
    #
    # parse cluster path
    #
    print('=> parsing cluster path...')
    cluster_path = dict()
    num_lines_loaded = 0
    path_len = 0
    for line in open(cluster_path_file).read().splitlines():
        if not line:
            continue
        cluster, word, frequency = line.split('\t')

        cluster_path[f(word)] = cluster

        if len(cluster) > path_len:
            path_len = len(cluster)

        num_lines_loaded += 1
        if num_lines_loaded % 50 == 0:
            sys.stdout.write("%d word paths loaded.\r" % num_lines_loaded )
            sys.stdout.flush()

    # padding short paths
    for k, v in cluster_path.items():
        v += '0' * (path_len - len(v))
        cluster_path[k] = v

    print('=> %d words with cluster path are loaded in total.' %
          len(cluster_path))
    print('max path len is %d' % path_len)

    #
    # sentence input
    #
    sentence_clusters = []
    unk_cluster_path = ['0' * int(0.4*path_len),
                        '0' * int(0.6*path_len),
                        '0' * int(0.8*path_len),
                        '0' * path_len]
    cluster_coverage = collections.defaultdict(int)
    token_count = collections.defaultdict(int)
    token_with_labels = collections.defaultdict(int)
    token_with_labels_covered = collections.defaultdict(int)
    for i, s in enumerate(sentences):
        if i % 100 == 0:
            sys.stdout.write('%d sentences processed.\r' % i)
            sys.stdout.flush()

        s_cluster_path = []
        for token in s:
            text = f(token[0])
            label = token[-1]

            token_count[text] += 1
            if label != 'O':
                token_with_labels[text] += 1
            if text in cluster_path:
                c_path = cluster_path[text]
                cluster_coverage[text] += 1
                if label != 'O':
                    token_with_labels_covered[text] += 1
            else:
                c_path = '0' * path_len

            s_cluster_path.append([c_path[:int(0.4*path_len)],
                                   c_path[:int(0.6*path_len)],
                                   c_path[:int(0.8*path_len)],
                                   c_path])

        s = []
        # add prev and next word cluster path
        for j in range(len(s_cluster_path)):
            if j == 0:
                prev_cp = unk_cluster_path
            else:
                prev_cp = s_cluster_path[j-1]
            if j == len(s_cluster_path)-1:
                next_cp = unk_cluster_path
            else:
                next_cp = s_cluster_path[j+1]
            s.append(prev_cp + s_cluster_path[j] + next_cp)

        sentence_clusters.append(s)

    print('%d / %d (%.2f) tokens have clusters.' % (sum(cluster_coverage.values()),
                                                    sum(token_count.values()),
                                                    sum(cluster_coverage.values()) / sum(token_count.values())))
    print('%d / %d (%.2f) unique tokens have clusters.' % (len(cluster_coverage),
                                                           len(token_count),
                                                           len(cluster_coverage) / len(token_count)))

    print('%d / %d (%.2f) labeled tokens have clusters.' % (
        sum(token_with_labels_covered.values()),
        sum(token_with_labels.values()),
        sum(token_with_labels_covered.values()) / (sum(token_with_labels.values())+1)
    ))
    print('%d / %d (%.2f) labeled unique tokens have clusters.' % (
        len(token_with_labels_covered),
        len(token_with_labels),
        len(token_with_labels_covered) / (len(token_with_labels)+1)
    ))

    return sentence_clusters


#
# postagging features
#
def postag_feature(sentences, parameters):
    print('=> loading pos features...')
    postag = load_sentences(parameters['pos'])
    assert len(sentences) == len(postag), 'number of sentences do not match.'
    rtn = []
    for i, s in enumerate(postag):
        assert len(s) == len(sentences[i]), 'number of tokens do not match in %d' % i
        t = [[w[-1]] for w in s]
        rtn.append(t)

    return rtn


def merge_features(sent_features):
    res = []
    for s_feat in zip(*sent_features):
        merged_sent_f = []
        for t_feat in zip(*s_feat):
            merged_t_feat = list(itertools.chain.from_iterable(t_feat))
            merged_sent_f.append(merged_t_feat)
        res.append(merged_sent_f)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bio_input',
                        help='bio file that needs to generate features')
    parser.add_argument('bio_output',
                        help='path of the result')
    parser.add_argument('--feat_column', type=int, default=1, required=True,
                        help='the number of the column where the features '
                             'start. default is 1, the 2nd column.')
    #
    # external features
    #
    parser.add_argument(
        "--pos", default="",
        help="path of pos result."
    )
    parser.add_argument(
        "--cluster", default="",
        help="path of brown cluster paths."
    )
    args = parser.parse_args()

    # external features
    parameters = dict()
    parameters['pos'] = args.pos
    parameters['cluster'] = args.cluster

    sentences = load_sentences(args.bio_input)

    feats = generate_features(sentences, parameters)

    # output bio with features
    if feats:
        bio = []
        for i, s in enumerate(sentences):
            bio_s = []
            for j, w in enumerate(s):
                bio_s.append(' '.join(w[:args.feat_column] + feats[i][j] +
                                      w[args.feat_column:]))
            bio.append('\n'.join(bio_s))
        with open(args.bio_output, 'w') as f:
            f.write('\n\n'.join(bio))
    else:
        shutil.copy(args.bio_input, args.bio_output)


