"""
process for lfw dataset
"""
import os

parent_dir = '../joint_bayesian/'
data_url = 'data/lfw_r/'
intra_pair_list = open(parent_dir+'data/intra_pairs.txt', 'r')
extra_pair_list = open(parent_dir+'data/extra_pairs.txt', 'r')

intra_pair_list_path = open(parent_dir+'data/intra_pairs_path.txt', 'w')
extra_pair_list_path = open(parent_dir+'data/extra_pairs_path.txt', 'w')

dirs = os.listdir(parent_dir+data_url)

while True:
    line = intra_pair_list.readline()
    if line:
        line = line.split()
        c_path = parent_dir+data_url+line[0]
        images = sorted(os.listdir(c_path))
        for i in images:
            if line[1] in i:
                img1 = c_path+'/'+i
            if line[2] in i:
                img2 = c_path+'/'+i
        intra_pair_list_path.write(img1+' '+img2+'\n')
    else:
        break
intra_pair_list.close()
intra_pair_list_path.close()

while True:
    line = extra_pair_list.readline()
    if line:
        line = line.split()
        c_path1 = parent_dir+data_url+line[0]
        c_path2 = parent_dir+data_url+line[2]
        images1 = sorted(os.listdir(c_path1))
        images2 = sorted(os.listdir(c_path2))
        for i in images1:
            if line[1] in i:
                img1 = c_path1+'/'+i
        for j in images2:
            if line[3] in j:
                img2 = c_path2+'/'+j
        extra_pair_list_path.write(img1+' '+img2+'\n')
    else:
        break
extra_pair_list.close()
extra_pair_list_path.close()
