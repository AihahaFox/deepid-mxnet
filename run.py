import mxnet as mx
import os
import argparse
import logging
import numpy as np
from tools import tools
from joint_bayesian import jb_model
from conv_net import symbol_deepid1
from util import align_face
import random
import time
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
        default='./model/deep3-0', help='the prefix of deepid')
parser.add_argument(
        '--epoch', type=int,
        default='110', help='the epoch of deepid')
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
        '--people', type=str, help='the sample path')
a_args = parser.parse_args()


def deepid(sample_path, predict):
    # align for image
    out_url = './util/data/'
    sample = align_face.alignMain(sample_path, out_url, predict)
    if sample == "Unable_align":
        return sample, None
    else:
        sample = tools.load_sample(sample)

    # load model from pre-trained model
    model = mx.model.FeedForward.load(
        prefix=a_args.prefix,
        epoch=a_args.epoch,
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

    #exact sample_feature
    sample_feature = feature_extractor.predict(sample)

    # image for own class
    people = os.listdir('./people_image/')
    people_feature = []
    people_label = []
    if len(people) == 0:
        tools.save_image(sample_path, './people_image/x_'+str(len(people)), sample_feature)
        return 'x'
    else:
        for i in people:
            path = './people_image/'+i
            img = os.listdir(path)
            random.shuffle(img)
            k = 0
            # load feature from file
            for j in img:
                if '.npy' in j and k < 5:
                    k += 1
                    people_feature.append(np.load(path+'/'+j))
                    people_label.append(i)

        # calculate the ratio of these images
        A, G = tools.load_A_G(a_args.A, a_args.G)
        people_ratio = [jb_model.Verify(A, G, sample_feature, people_feature[i]) for i in xrange(len(people_label))]
        people_result = tools.predict_face(people_ratio, 13, people_label)
        if people_result.count(0) == len(people_result):
            return 'unknow_person', sample_feature
        else:
            dic = {i: people_result.count(i) for i in people}
            fname = max(dic, key=dic.get)
            return fname, sample_feature
