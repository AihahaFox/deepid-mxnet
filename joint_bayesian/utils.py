import cPickle as pickle
import time
from sklearn.decomposition import PCA
from sklearn.externals import joblib


# time
def get_time_str():
    return time.strftime("%Y-%m-%d, %H:%M:%S ", time.localtime((time.time())))


# print informaion with time
def print_info(msg):
    print get_time_str(), msg


# saving data into pkl
def data_to_pkl(data, file_path):
    print "Saving data to file(%s). " % (file_path)

    with open(file_path, "w") as f:
        pickle.dump(data, f)
        return True

    print "Occur Error while saving..."
    return False


def read_pkl(file_path):
    with open(file_path, "r") as f:
        return pickle.load(f)


# PCA train model
def PCA_Train(data, result_fold, n_components=2000):
    print_info("PCA training (n_components=%d)..." % n_components)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    joblib.dump(pca, result_fold + "pca_model.m")

    print_info("PCA done.")

    return pca
