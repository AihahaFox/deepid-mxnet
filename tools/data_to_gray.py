"""
script for transforming data to gray mode
"""
import skimage.io as io
import skimage.color as color
import skimage.transform as transform
import numpy as np
import os

# raw data
url = '../joint_bayesian/data/lfw/'
# result data
url2 = '../joint_bayesian/data/lfw_r/'

if not os.path.isdir(url2):
    os.mkdir(url2)

classes = sorted(os.listdir(url))
for c in classes:
    c_path = url + c + '/'
    new_path = url2 + c + '/'
    os.mkdir(new_path)
    imgs = os.listdir(c_path)
    for i in imgs:
        img = io.imread(c_path + i)
        new_img = transform.resize(img, (55, 55, 3))
        io.imsave(new_path+i, new_img)
    print c + 'done!'
