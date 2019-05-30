import argparse


def merge_mono_word_emb(emb1, lang1, emb2, lang2, merged_file):
    out = open(merged_file, 'w')
    with open(emb1, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts)>5:
                out.write(lang1+":"+line)
    with open(emb2, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts)>5:
                out.write(lang2+":"+line)

    out.close()


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # here list the parameters for optimizing the model
    parser.add_argument("--emb1", type=str, default='')
    parser.add_argument("--lang1", type=str, default='')
    parser.add_argument("--emb2", type=str, default='')
    parser.add_argument("--lang2", type=str, default='')
    parser.add_argument("--result", type=str, default='')

    args = parser.parse_args()
    emb1 = args.emb1
    lang1 = args.lang1
    emb2 = args.emb2
    lang2 = args.lang2
    merged_file = args.result

    merge_mono_word_emb(emb1, lang1, emb2, lang2, merged_file)