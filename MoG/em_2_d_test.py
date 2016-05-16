import math
import copy
import numpy as np
import matplotlib.pyplot as plt


# given the k parameters of Gaussian distribution
# here we make k=2, note that this two Gaussian distributions have the same sigma, but the different mu.
def ini_data(Sigma,Mu1,Mu2,k,N):
    global X
    global Mu
    global Expectations
    X = np.zeros((1,N))
    Mu = np.random.random(2)
    Expectations = np.zeros((N,k))
    for i in xrange(0,N):
        if np.random.random(1) > 0.5:
            X[0,i] = np.random.normal()*Sigma + Mu1
        else:
            X[0,i] = np.random.normal()*Sigma + Mu2


# EM Algorithm: The first step--compute E[zij]
def e_step(Sigma,k,N):
    global Expectations
    global Mu
    global X
    for i in xrange(0,N):
        Denom = 0
        for j in xrange(0,k):
            Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
        for j in xrange(0,k):
            Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
            Expectations[i,j] = Numer / Denom


# EM Algotirhm: The second step--compute the parameter that makes the E[zij] is the max
def m_step(k,N):
    global Expectations
    global X
    for j in xrange(0,k):
        Numer = 0
        Denom = 0
        for i in xrange(0,N):
            Numer += Expectations[i,j]*X[0,i]
            Denom +=Expectations[i,j]
        Mu[j] = Numer / Denom


# the algorithm iter about iter_num, or reach the precision of Epsilon, then the iteration is stopped
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):
    ini_data(Sigma,Mu1,Mu2,k,N)
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        e_step(Sigma,k,N)
        m_step(k,N)
        print "iter num is " + str(i) + " and the Mu is ", Mu
        # print i,Mu
        if sum(abs(Mu-Old_Mu)) < Epsilon:
            break


if __name__ == '__main__':
    run(6, 40, 20, 2, 1000, 1000, 0.0001)
    plt.hist(X[0, :], 50)
    plt.show()