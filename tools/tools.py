from skimage import io
import os
import numpy as np
import cPickle as pickle
import mxnet as mx
import random

url = './data/casia_all_train.bin'
mean = mx.nd.load(url)['mean_img'].asnumpy()

def load_sample(people_path):
    sample = io.imread(people_path)
    sample = np.resize(sample, (55, 55, 3))
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    sample = sample - mean
    sample = sample / 255.0
    sample = np.resize(sample, (1, 3, 55, 55))
    return sample

def load_path_from_name(name, classes, data_url='./data/CASIA_RESIZE/'):
    url = data_url + name +'/'
    raw = os.listdir(url)
    random.shuffle(raw)
    if classes == 0:
        path = [url+raw[i] for i in range(5)]
    else:
        path = [url+raw[i] for i in range(len(raw)) if i < 5]
    return path

def img_process(img_path_list):
    img_list = []
    for i in img_path_list:
        sample = io.imread(i)
        sample = np.resize(sample, (55, 55, 3))
        # sample = sample / 256.0
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)

        sample = sample - mean
        sample = sample / 255.0


        img_list.append(sample)
    img_list = np.resize(img_list, (len(img_path_list), 3, 55, 55))
    return img_list


def load_A_G(A_path, G_path):
    with open(A_path, "rb") as f:
        A = pickle.load(f)
    with open(G_path, "rb") as f:
        G = pickle.load(f)
    return A, G


def predict_face(ratio_list, thread, label_list):
    result = []
    for i in list(ratio_list):
        if i < thread:
            result.append(0)
        else:
            result.append(label_list[ratio_list.index(i)])
    return result


def save_image(sample_path, path, feature):
    if not os.path.isdir(path):
        os.mkdir(path)
    c = os.listdir(path)
    np.save(path+'/'+str(len(c)/2), feature)
    io.imsave(os.path.join(path, str(len(c)/2)+'.jpg'), io.imread(sample_path))
