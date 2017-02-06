import mxnet as mx
import argparse
import train_model

parser = argparse.ArgumentParser(
        description='train an image deepid network')
parser.add_argument(
        '--network', type=str,
        default='deepid1', help='the cnn to use')
parser.add_argument(
        '--data-dir', type=str,
        default='../data/', help='the input data directory')
parser.add_argument(
        '--gpus', type=str,
        default='0', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument(
        '--num-examples', type=int,
        default=293490, help='the number of training examples')
parser.add_argument(
        '--batch-size', type=int,
        default=256, help='the batch size')
parser.add_argument(
        '--lr', type=float,
        default=0.001, help='the initial learning rate')
parser.add_argument(
        '--lr-factor', type=float,
        default=1, help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument(
        '--lr-factor-epoch', type=float,
        default=1, help='the number of epoch to factor the lr, could be .5')
parser.add_argument(
        '--model-prefix', type=str,
        help='the prefix of the model to load')
parser.add_argument(
        '--save-model-prefix', type=str,
        help='the prefix of the model to save')
parser.add_argument(
        '--num-epochs', type=int,
        default=600, help='the number of training epochs')
parser.add_argument(
        '--load-epoch', type=int,
        help="load the model on an epoch using the model-prefix")
parser.add_argument(
        '--kv-store', type=str,
        default='local', help='the kvstore type')
args = parser.parse_args()

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(8873)


# data
def get_iterator(args, kv):
    data_shape = (3, 55, 55)

    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + "casia_raw_train.rec",
        mean_img=args.data_dir + "casia_raw_train.bin",
        scale=1./255,
        data_shape=data_shape,
        batch_size=args.batch_size,
        rand_crop=False,
        rand_mirror=True,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + "casia_raw_val.rec",
        mean_img=args.data_dir + "casia_raw_val.bin",
        scale=1./255,
        rand_crop=False,
        rand_mirror=False,
        data_shape=data_shape,
        batch_size=args.batch_size,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
