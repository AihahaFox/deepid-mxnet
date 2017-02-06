import mxnet as mx
import argparse
import logging
import numpy as np
from tools import tools
from joint_bayesian import jb_model
from conv_net import symbol_deepid1
import matplotlib.pyplot as plt
# network
net = symbol_deepid1.get_symbol(8873)

# logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# argparse
parser = argparse.ArgumentParser(
        description='test an image on the pre-trained deepid network')
parser.add_argument(
        '--prefix', type=str,
        default='./model/deep4-0', help='the prefix of deepid')
parser.add_argument(
        '--epoch', type=int,
        default='125', help='the epoch of deepid')
parser.add_argument(
        '--gpus', type=str,
        default='0', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument(
        '--result-fold', type=str,
        default='./result/', help='result fold')
parser.add_argument(
        '--A', type=str,
        default='./result/A.pkl', help='A')
parser.add_argument(
        '--G', type=str,
        default='./result/G.pkl', help='G')
parser.add_argument(
        '--predict-classes', type=bool,
        default=True, help='wheither to predict the class')
parser.add_argument(
        '--extract-feature', type=bool,
        default=True, help='wheither to extract the feature')
parser.add_argument(
        '--people', type=str,
        help='the sample path')
args = parser.parse_args()

# cpu or gpu
devs = mx.cpu() if args.gpus is None else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]
# load model from pre-trained model
model = mx.model.FeedForward.load(
        prefix=args.prefix,
        epoch=args.epoch,
        ctx=mx.gpu(),
        numpy_batch_size=1)

# create feature_extractor for deepid
internals = model.symbol.get_internals()
# get feature layer symbol out of internals
fea_symbol = internals["fc3_output"]
feature_extractor = mx.model.FeedForward(
        ctx=mx.gpu(),
        symbol=fea_symbol,
        numpy_batch_size=1,
        arg_params=model.arg_params,
        aux_params=model.aux_params,
        allow_extra_params=True)
# prepare for the image
url1 = '/home/galaxyeye-tech/docs/deepid/joint_bayesian/data/extra_pairs_path.txt'
url2 = '/home/galaxyeye-tech/docs/deepid/joint_bayesian/data/intra_pairs_path.txt'

ratio_list1 = []
ratio_list2 = []
A, G = tools.load_A_G(args.A, args.G)


def cos_dis(f1, f2):
    a = 0
    b = 0
    c = 0
    for i in range(len(f1)):
        a += f1[i]*f2[i]
        b += f1[i]**2
        c += f2[i]**2
    return a/((b**0.5)*(c**0.5))

f1 = open(url1, 'r')
f2 = open(url2, 'r')
# extract feature for extra pairs
while True:
    line = f1.readline()
    if line:
        line = line.split()
        img_list = tools.img_process(line)
        img_feature = feature_extractor.predict(img_list)
        # img_feature = jb_model.data_pre(img_feature)
        ratio_list1.append(
             cos_dis(img_feature[0], img_feature[1]))
             # jb_model.Verify(A, G, img_feature[0], img_feature[1]))
    else:
        break
# print img_feature[0]
# extract feature for intra pairs
while True:
    line = f2.readline()
    if line:
        line = line.split()
        img_list = tools.img_process(line)
        img_feature = feature_extractor.predict(img_list)
        # img_feature = jb_model.data_pre(img_feature)
        ratio_list2.append(
             cos_dis(img_feature[0], img_feature[1]))
             # jb_model.Verify(A, G, img_feature[0], img_feature[1]))
    else:
        break

plt.subplot(1, 2, 1)
plt.hist(ratio_list1)
plt.subplot(1, 2, 2)
plt.hist(ratio_list2)
plt.show()

for k in np.arange(0.0, 0.4, 0.01):
    label1 = []
    label2 = []
    for i in ratio_list1:
        if i > k:
            label1.append(1)
        else:
            label1.append(0)
    for i in ratio_list2:
        if i > k:
            label2.append(1)
        else:
            label2.append(0)
    count = 0
    for i in range(2979):
        if label1[i] == 0:
            count += 1
    for i in range(2992):
        if label2[i] == 1:
            count += 1
    print 'threshold=%f, p=%f, count=%d' % (k, float(count) / (2979+2992), count)
