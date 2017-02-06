"""
script for saving data to .mat
"""
import os
from scipy import io
from skimage import io as skio
import numpy as np

data_url = '/home/galaxyeye-tech/docs/deepid/data/CASIA2/'
labels = os.listdir(data_url)
print labels
data = []
label = []
for l in labels:
    path = data_url + l + '/'
    sub_imgs = os.listdir(path)
    for i in sub_imgs:
        img_path = skio.imread(path + i)
        img_path = np.resize(img_path, (55, 55, 3))
        data.append(img_path)
        label.append(l)
data = np.asarray(data)
label = np.asarray(label)
url = '/home/galaxyeye-tech/docs/deepid/data/JB_RAW_train.mat'
io.savemat(url, {'data': data, 'label': label})
