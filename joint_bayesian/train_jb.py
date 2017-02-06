"""
train the joint bayesian model
"""
# coding=utf-8
import argparse
from scipy.io import loadmat
from sklearn.externals import joblib
from joint_bayesian import jb_model, utils
import numpy as np
import mxnet as mx
from conv_net import symbol_deepid1

# network
net = symbol_deepid1.get_symbol(7873)

parser = argparse.ArgumentParser(
        description='train joint bayesian model')
parser.add_argument(
        '--trainset', type=str,
        default='../data/JB_OWN_train.mat', help='the trainset to use')
parser.add_argument(
        '--prefix', type=str,
        default='../model/deep2-0', help='the prefix of deepid')
parser.add_argument(
        '--epoch', type=int,
        default='125', help='the epoch of deepid')
parser.add_argument(
        '--result_fold', type=str,
        default='../data/', help='the output data directory')
parser.add_argument(
        '--pca_dimension', type=int,
        default='160', help='the dimention pca to process')
parser.add_argument(
        '--pca_train', type=bool,
        default=False, help='the pretrained pca model')
parser.add_argument(
        '--deep_train', type=bool,
        default=True, help='the pretrained pca model')
parser.add_argument(
        '--pretrained_pca_model', type=bool,
        default=False, help='the pretrained pca model')
args = parser.parse_args()

# load data
print 'loading data...'
data = loadmat(args.trainset)['data']
label = loadmat(args.trainset)['label']

# pca train
if args.pca_train:
    if args.pretrained_pca_model:
        # use pre_trained model to tranform data
        clt_pca = joblib.load(args.result_fold + "pca_model.m")
        pca_data = clt_pca.transform(data)
    else:
        # train a pca model
        pca = utils.PCA_Train(data, args.result_fold, args.pca_dimension)
        sample_data = pca.transform(data)

# load mean from convnet trainset
url = '../data/casia_resize_train.bin'
mean = mx.nd.load(url)['mean_img'].asnumpy()

# deep train
if args.deep_train:
    # load model from pre-trained model
    model = mx.model.FeedForward.load(
            prefix=args.prefix,
            epoch=args.epoch,
            ctx=mx.gpu(),
            numpy_batch_size=1)

    # create feature_extractor for deepid
    internals = model.symbol.get_internals()
    # get feature layer symbol out of internals
    fea_symbol = internals["fc1_output"]
    feature_extractor = mx.model.FeedForward(
            ctx=mx.gpu(),
            symbol=fea_symbol,
            numpy_batch_size=1,
            arg_params=model.arg_params,
            aux_params=model.aux_params,
            allow_extra_params=True)
    new_data = []
    # process data to extract feature
    for i in data:
        i = np.swapaxes(i, 0, 2)
        i = np.swapaxes(i, 1, 2)
        i = i - mean
        i = i / 255.0
        new_data.append(i)
    new_data = np.resize(new_data, (data.shape[0], 3, 55, 55))
    print new_data[0]
    sample_data = feature_extractor.predict(new_data)

# train joint bayesian model
jb_model.JointBayesian_Train(sample_data, label, args.result_fold)
