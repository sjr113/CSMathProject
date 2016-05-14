#################################################
# SVM: support vector machine
# Author : zouxy
# Date   : 2013-12-12
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com

# http://blog.csdn.net/zouxy09/article/details/17292011
################################################

import numpy as np
import svm_2_d


def test_on_svm():
    ################## test svm #####################
    ## step 1: load data
    print "step 1: load data..."
    dataSet = []
    labels = []
    N = 100
    dataSet, labels = generate_two_dimension_data(N)
    # fileIn = open('E:/Python/Machine Learning in Action/testSet.txt')
    # for line in fileIn.readlines():
    #     lineArr = line.strip().split('\t')
    #     dataSet.append([float(lineArr[0]), float(lineArr[1])])
    #     labels.append(float(lineArr[2]))
    #
    # dataSet = np.mat(dataSet)
    # labels = np.mat(labels).T
    train_x = dataSet[0:81, :]
    train_y = labels[0:81, :]
    test_x = dataSet[80:101, :]
    test_y = labels[80:101, :]

    ## step 2: training...
    print "step 2: training..."
    C = 0.6
    toler = 0.001
    maxIter = 50
    svmClassifier = svm_2_d.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

    ## step 3: testing
    print "step 3: testing..."
    accuracy = svm_2_d.testSVM(svmClassifier, test_x, test_y)

    ## step 4: show the result
    print "step 4: show the result..."
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    svm_2_d.showSVM(svmClassifier, dataSet, labels)


def generate_two_dimension_data(N):
    X = np.vstack([np.random.rand(N, 2), np.random.rand(N, 2)+2])
    y = np.vstack([np.tile([0], [N, 1]), np.tile([1], [N, 1])])
    return X, y


if __name__ == "__main__":
    test_on_svm()

