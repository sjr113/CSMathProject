import numpy as np
import matplotlib.pyplot as plt
import mixture_of_Gaussian


def generate_two_d_gaussian_data():
    pass
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()
    return x, y


def ini_data(Sigma, Mu1, Mu2, k, N):
    # initial the data set
    # the data has the same mean value but a different variance
    X = np.zeros((N, 1))
    Mu = np.random.random(2)
    Expectations = np.zeros((N, k))
    for i in xrange(0, N):
        if np.random.random(1) > 0.5:
            X[i, 0] = np.random.normal()*Sigma + Mu1
        else:
            X[i, 0] = np.random.normal()*Sigma + Mu2

    return X


if __name__ == "__main__":
    # x, y = generate_two_d_gaussian_data()

    X = ini_data(6, 40, 20, 2, 1000)
    # visualize the data set
    # plt.figure()
    # plt.hist(X[:, 0],50)
    # plt.show()

    data_mat, px, pMiu, pPi, pSigma = mixture_of_Gaussian.gmm(X, 2)

    print pMiu
