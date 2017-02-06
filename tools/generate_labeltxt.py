"""
genrate label.txt for dataset
"""
import os

url = '../data/CASIA_GRAY/'
f = open('../data/label.txt', 'w')
c = os.listdir(url)
for i in c:
    f.write(i+'\n')
f.close
