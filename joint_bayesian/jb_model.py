# coding=utf-8
import numpy as np
import utils
import gnumpy as gpu
import time

# GPU cov function
def cov(x):
    y = gpu.mean(x, axis=1)[:, None]
    x = x.as_numpy_array().__sub__(y.as_numpy_array())
    x_T = x.T.conj()
    result = gpu.dot(x, x_T)
    result = result.__div__(x.shape[1] - 1)
    return result


# Before training,the mean must be substract
def JointBayesian_Train(trainingset, label, fold="./"):
    if fold[-1] != '/':
        fold += '/'
    print trainingset.shape
    print trainingset[0]
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # print classes
    # print labels
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image, np.float32)
    maxNumberInOneClass = 0
    for i in range(n_class):
        # get the item of i
        cur[i] = trainingset[labels == i]  # 00x
        # cur_gpu = shared(cur[i])
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]

        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    utils.print_info("prepare done, maxNumberInOneClass=" +
                     str(maxNumberInOneClass))

    u = np.zeros([n_dim, n_class], np.float32)
    u_gpu = gpu.garray(u)
    ep = np.zeros([n_dim, withinCount], np.float32)
    ep_gpu = gpu.garray(ep)
    nowp = 0
    for i in range(n_class):
        # the mean of cur[i]
        cur_gpu = gpu.garray(cur[i])
        u_gpu[:, i] = gpu.mean(cur_gpu, 0)
        b = u_gpu[:, i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep_gpu[:, nowp:nowp + n_same_label] = cur_gpu.T - b
            nowp += n_same_label
    utils.print_info("stage1 done")

    Su = cov(u_gpu)
    gpu.status()
    Sw = cov(ep_gpu)
    oldSw = Sw
    SuFG = {}
    SwG = {}
    convergence = 1
    min_convergence = 1
    for l in range(500):
        F = np.linalg.pinv(Sw.as_numpy_array())
        F_gpu = gpu.garray(F)
        u = np.zeros([n_dim, n_class], np.float32)
        u_gpu = gpu.garray(u)
        ep = np.zeros([n_dim, n_image], np.float32)
        ep_gpu = gpu.garray(ep)
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
                # G = −(mS μ + S ε )−1*Su*Sw−1
                temp = np.linalg.pinv(
                    mi * Su.as_numpy_array() + Sw.as_numpy_array())
                temp2 = gpu.dot(gpu.garray(temp), Su)
                G = -gpu.dot(temp2, F_gpu)
                # Su*(F+mi*G) for u
                SuFG[mi] = gpu.dot(Su, (F_gpu + mi * G))
                # Sw*G for e
                SwG[mi] = gpu.dot(Sw, G)
        utils.print_info('stage2 done')
        # print SuFG
        for i in range(n_class):
            # print l, i
            nn_class = cur[i].shape[0]
            # print nn_class
            cur_gpu = gpu.garray(cur[i])
            # formula 7 in suppl_760
            temp = gpu.dot(SuFG[nn_class], cur_gpu.T)
            u_gpu[:, i] = gpu.sum(temp, 1)
            # formula 8 in suppl_760
            ep_gpu[:, nowp:nowp + nn_class] = cur_gpu.T + \
                gpu.sum(gpu.dot(SwG[nn_class], cur_gpu.T), 1).reshape(n_dim, 1)
            nowp = nowp + nn_class
        print 'stage2 done'

        Su = cov(u_gpu)
        Sw = cov(ep_gpu)
        convergence = np.linalg.norm(
            (Sw - oldSw).as_numpy_array()) / np.linalg.norm(Sw.as_numpy_array())
        utils.print_info("Iterations-" + str(l) + ": " + str(convergence))
        if convergence < 1e-6:
            print "Convergence: ", l, convergence
            break
        oldSw = Sw

        if convergence < min_convergence:
            min_convergence = convergence
            F = np.linalg.pinv(Sw.as_numpy_array())
            F_gpu = gpu.garray(F)
            G = -gpu.dot(gpu.dot(np.linalg.pinv((2 * Su + Sw).as_numpy_array()), Su.as_numpy_array()), F_gpu)
            A = np.linalg.pinv((Su + Sw).as_numpy_array()) - \
                (F + G.as_numpy_array())
            utils.data_to_pkl(G, fold + "G.pkl")
            utils.data_to_pkl(A, fold + "A.pkl")

    F = np.linalg.pinv(Sw.as_numpy_array())
    F_gpu = gpu.garray(F)
    temp = gpu.garray(np.linalg.pinv((2 * Su + Sw).as_numpy_array()))
    G = -gpu.dot(gpu.dot(temp, Su), F_gpu).as_numpy_array()
    A = np.linalg.pinv((Su + Sw).as_numpy_array()) - (F + G)
    utils.data_to_pkl(G, fold + "G_con.pkl")
    utils.data_to_pkl(A, fold + "A_con.pkl")

    return A, G


# ratio of similar,the threshold
def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = gpu.dot(gpu.dot(x1.T, A), x1) + gpu.dot(gpu.dot(
        x2.T, A), x2) - 2 * gpu.dot(gpu.dot(x1.T, G), x2)
    ratio = ratio.as_numpy_array()
    return float(ratio)
