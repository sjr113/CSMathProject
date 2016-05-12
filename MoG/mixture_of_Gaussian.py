#coding=utf-8

import numpy as np
import pylab
import random, math


# load the dataset
def load_data_set(file_name):
    # general function to parse tab -delimited floats
    data_mat = []
    # assume last column is target value
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)
        # map all elements to float()
        data_mat.append(flt_line)
    return data_mat


# def gmm(file, K_or_centroids):
def gmm(data_mat, K_or_centroids):
    # Expectation-Maximization iteration implementation of
    # Gaussian Mixture Model.
    #
    # PX = GMM(X, K_OR_CENTROIDS)
    # [PX MODEL] = GMM(X, K_OR_CENTROIDS)
    #
    #  - X: N-by-D data matrix.
    #  - K_OR_CENTROIDS: either K indicating the number of
    #       components or a K-by-D matrix indicating the
    #       choosing of the initial K centroids.
    #
    #  - PX: N-by-K matrix indicating the probability of each
    #       component generating each point.
    #  - MODEL: a structure containing the parameters for a GMM:
    #       MODEL.Miu: a K-by-D matrix.
    #       MODEL.Sigma: a D-by-D-by-K matrix.
    #       MODEL.Pi: a 1-by-K vector.

    # Generate Initial Centroids
    threshold = 1e-15
    # data_mat = mat(load_data_set(file))

    [N, D] = np.shape(data_mat)
    # K_or_centroids = 2
    # K_or_centroids可以是一个整数，也可以是k个质心的二维列向量
    if np.shape(K_or_centroids)==():   # if K_or_centroid is a 1*1 number
        K = K_or_centroids
        Rn_index = range(N)
        random.shuffle(Rn_index)  # random index N samples
        centroids = data_mat[Rn_index[0:K], :]   # generate K random centroid
    else: # K_or_centroid is a initial K centroid
        K = np.size(K_or_centroids)[0]
        centroids = K_or_centroids

    # initial values
    [pMiu, pPi, pSigma] = init_params(data_mat,centroids,K,N,D)
    Lprev = -np.inf # 上一次聚类的误差

    # EM Algorithm
    while True:
        # Estimation Step
        Px = calc_prob(pMiu,pSigma,data_mat,K,N,D)

        # new value for pGamma(N*k), pGamma(i,k) = Xi由第k个Gaussian生成的概率
        # 或者说xi中有pGamma(i,k)是由第k个Gaussian生成的
        pGamma = np.mat(np.array(Px) * np.array(np.tile(pPi, [N, 1])))  # 分子 = pi(k) * N(xi | pMiu(k), pSigma(k))
        pGamma = pGamma / (np.tile(np.sum(pGamma, 1), [1, K]) + 1e-6)  # 分母 = pi(j) * N(xi | pMiu(j), pSigma(j))对所有j求和

        # Maximization Step - through Maximize likelihood Estimation
        # print 'dtypeddddddddd:',pGamma.dtype
        Nk = np.sum(pGamma, 0)  # Nk(1*k) = 第k个高斯生成每个样本的概率的和，所有Nk的总和为N。

        # update pMiu

        pMiu = np.mat(np.diag((1/(Nk+1e-6)).tolist()[0])) * (pGamma.T) * data_mat  # update pMiu through MLE(通过令导数 = 0得到)
        pPi = Nk/N

        # update k个 pSigma

        for kk in range(K):
            Xshift = data_mat-np.tile(pMiu[kk], (N, 1))

            Xshift.T * np.mat(np.diag(pGamma[:, kk].T.tolist()[0])) * Xshift / 2

            pSigma[:, :, kk] = (Xshift.T * \
                np.mat(np.diag(pGamma[:, kk].T.tolist()[0])) * Xshift) / (Nk[0, kk] + 1e-6)

        # check for convergence
        L = sum(np.log(Px*(pPi.T)+1e-6))
        if L-Lprev < threshold:
            break
        Lprev = L

    return data_mat, Px, pMiu, pPi, pSigma


def init_params(X,centroids,K,N,D):
    pMiu = centroids  # k*D, 即k类的中心点
    pPi = np.zeros([1, K])  # k类GMM所占权重（influence factor）
    pSigma = np.zeros([D, D, K])  # k类GMM的协方差矩阵，每个是D*D的

    # 距离矩阵，计算N*K的矩阵（x-pMiu）^2 = x^2+pMiu^2-2*x*Miu
    # x^2, N*1的矩阵replicateK列\#pMiu^2，1*K的矩阵replicateN行

    distmat = np.tile(np.reshape(np.sum(np.power(X, 2), 1), [N, 1]), [1, K]) + \
        np.tile(np.reshape(np.transpose(np.sum(np.power(pMiu,2), 1)), [1, K]), [N, 1]) -  \
        2*X*np.transpose(pMiu)
    labels = distmat.argmin(1)  # Return the minimum from each row

    # 获取k类的pPi和协方差矩阵

    for k in range(K):
        indexList = []
        boolList = (labels==k).tolist()
        # indexList = [boolList.index(i) for i in boolList if i==[True]]

        for i in range(len(boolList)):
            if boolList[i]==True:
                indexList.append(i)

        Xk = X[indexList, :]
        print
        # print cov(Xk)
        # 也可以用shape(XK)[0]
        pPi[0][k] = float(np.size(Xk, 0))/N
        pSigma[:, :, k] = np.cov(np.transpose(Xk))

    return pMiu,pPi,pSigma


# 计算每个数据由第k类生成的概率矩阵Px
def calc_prob(pMiu, pSigma, X, K, N, D):
    # Gaussian posterior probability
    # N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))
    Px = np.mat(np.zeros([N, K]))
    for k in range(K):
        Xshift = X-np.tile(pMiu[k, :], [N, 1]) #X-pMiu
        # inv_pSigma = mat(pSigma[:, :, k]).I
        inv_pSigma = np.linalg.pinv(np.mat(pSigma[:, :, k]))

        tmp = np.sum(np.array((Xshift*inv_pSigma)) * np.array(Xshift), 1) # 这里应变为一列数
        tmp = np.mat(tmp).T
        # print linalg.det(inv_pSigma),'54545'

        Sigema = np.linalg.det(np.mat(inv_pSigma))

        if Sigema < 0:
            Sigema=0

        coef = np.power((2*(math.pi)),(-D/2)) * np.sqrt(Sigema)
        Px[:, k] = coef * np.exp(-0.5*tmp)
    return Px

# compute the accuracy of the data set, but there must be samples and labels
# def cal_accuracy(gnd, label):
#     # res = bestmap(gnd, label)
#     # acc = length(find(gnd == res))/len(gnd)
#     count = 0
#     for i in range(len(gnd)):
#         if gnd[i] == label[i]:
#             count += 1
#
#     acc = count / len(gnd)
#
#
# def cmp_accuracy(data, label, k):
#     #  calculate the accuracy clustered by GMM model
#     px = gmm(data, k)
#     [temp, ind] = max(px, [], 1)   # ind = cluster label
#     accuracy = cal_accuracy(ind, label)























