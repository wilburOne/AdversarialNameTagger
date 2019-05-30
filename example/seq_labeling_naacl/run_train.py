import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='', help='path for train file')
parser.add_argument('--bi_train', type=str, default='', help='path for train file')
parser.add_argument('--dev', type=str, default='', help='path for dev file')
parser.add_argument('--test', type=str, default='', help='path for test file')
parser.add_argument('--target_lang', type=str, default='', help='path for embedding file')
parser.add_argument('--related_lang', type=str, default='', help='path for embedding file')
parser.add_argument('--embedding', type=str, default='', help='path for embedding file')
parser.add_argument('--bi_embedding', type=str, default='', help='path for embedding file')
parser.add_argument('--model_dir', type=str, default='', help='path for model file')
parser.add_argument('--dico_eval', type=str, default='', help='path for eval word adversarial')
parser.add_argument('--dis_most_frequent', type=int, default='1', help='')
parser.add_argument('--cuda', type=int, default='1')
parser.add_argument('--signal', type=str, default="")

args = parser.parse_args()

train = args.train
bi_train = args.bi_train
dev = args.dev
test = args.test
target_lang = args.target_lang
related_lang = args.related_lang
pre_emb = args.embedding
bi_pre_emb = args.bi_embedding
cuda = args.cuda
signal = args.signal
model_dir = args.model_dir
dico_eval = args.dico_eval
dis_most_frequent = args.dis_most_frequent

# run command

script = '/data/m1/huangl7/Lorelei2018/name_tagging-master/dnn_pytorch/' \
         'seq_labeling_naacl/train.py'
cmd = [
    'python3',
    script,
    # data settings
    '--train', train,
    '--bi_train', bi_train,
    '--dev', dev,
    '--test', test,
    '--model_dp', model_dir,
    '--tag_scheme', 'iobes',

    # parameter settings
    '--lower', '0',
    '--zeros', '1',
    '--char_dim', '25',
    '--char_lstm_dim', '0',
    '--word_dim', '100',
    '--word_lstm_dim', '100',
    '--pre_emb', pre_emb,
    '--bi_pre_emb', bi_pre_emb,
    '--all_emb', '0',
    '--crf', '1',
    '--num_epochs', '100',
    '--lr_method', 'sgd-init_lr=.01-lr_decay_epoch=10',
    '--cuda', str(cuda),
    '--signal', signal,

    # parameters for adversarial
    '--target_lang', target_lang,
    '--related_lang', related_lang,
    '--max_vocab', '500000',
    '--dis_most_frequent', str(dis_most_frequent), #'100000',
    '--target_emb', pre_emb,
    '--related_emb', bi_pre_emb,
    '--dico_eval', dico_eval
]

# set OMP threads to 1
os.environ.update({'OMP_NUM_THREADS': '1'})
python_path = os.path.abspath(__file__).replace('example/seq_labeling_naacl/run_train.py', '')
os.environ.update({'PYTHONPATH': python_path})

print(' '.join(cmd))
subprocess.call(cmd, env=os.environ)
