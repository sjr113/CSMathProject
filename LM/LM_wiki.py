import numpy as np
from matplotlib import pyplot as plt


def LM_algorithm(ep, para_lambda, x0, y0, max_iter, num_data, num_para, data_set, obs_1):
    # the first step: given the initial values of parameters
    update = 1
    a_est = x0
    b_est = y0

    # the second step: the operation of iterations
    for iter_num in range(max_iter):
        if update == 1:
            # compute the Jacobi matrix
            J = np.zeros((num_data, num_para))
            for i in range(len(data_set)):
                J[i, :] = [test_my_fun_derv(b_est, a_est, data_set[i])]

            # compute the value of functions according to the current parameters
            y_est = test_my_fun(x0, y0, data_set)
            # compute the error
            d = obs_1 - y_est

            # compute the Hessian matrix
            H = np.dot(np.transpose(J), J)

            # if it is the first iter, compute the error
            if iter_num == 0:
                e = np.dot(d, np.transpose(d))

        # compute the matrix according to the para_lambda
        H_lm = H + para_lambda * np.eye(num_para, num_para)
        # compute the step dp
        dp = np.dot(np.linalg.inv(H_lm), (np.dot(np.transpose(J), np.reshape(d, [len(d), 1]))))
        # compute the error
        g = np.transpose(J) * d
        a_lm = a_est + dp[0]
        b_lm = b_est + dp[1]

        # compute the new value and error
        y_est_lm = test_my_fun(a_lm, b_lm, data_set)
        d_lm = obs_1 - y_est_lm
        e_lm = np.dot(d_lm, np.transpose(d_lm))

        # update the para_lambda according to the error
        if e_lm < e:
            if e_lm < ep:
                break
            else:
                para_lambda /= 5
                a_est = a_lm
                b_est = b_lm
                e = e_lm
                update = 1
        else:
            update = 0
            para_lambda *= 5
    return a_est, b_est


def test_my_fun_derv(b_est, a_est, data):
    derv_y = np.cos(b_est * data) + data * b_est * np.cos(a_est * data) - np.sin(b_est * data * a_est* data) \
             + np.sin(a_est * data)
    return derv_y


def test_my_fun(x0, y0, data1):
    y = x0 * np.cos(y0 * data1) + y0 * np.sin(x0 * data1)
    return y

def test_on_LM():
    data1 = [ 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,3.2, 3.4, 3.6, 3.8, 4.0,
              4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2]

    obs_1=[102.225, 99.815, -21.585, -35.099, 2.523, -38.865, -39.020, 89.147, 125.249, -63.405, -183.606, -11.287,
           197.627, 98.355, -131.977, -129.887, 52.596, 101.193, 5.412, -20.805, 6.549, -40.176, -71.425, 57.366,
           153.032, 5.301, -183.830, -84.612, 159.602, 155.021, -73.318, -146.955]

    # initial the start point
    x0 = 100
    y0 = 100

    # for example, we use the function of "x0 * np.cos(y0 * data1) + y0 * np.sin(x0 * data1)"
    y_init = test_my_fun(x0, y0, data1)

    # the number of samples
    num_data = len(obs_1)

    # the dimensions of the parameters
    num_para = 2

    # the max iterations of algorithm
    max_iters = 60

    # the initial value of lambda
    para_lambda = 0.1

    # the precision of the algorithm
    ep = 100

    data1 = np.array(data1)
    obs_1 = np.array(obs_1)

    a_est, b_est = LM_algorithm(ep, para_lambda, x0, y0, max_iters, num_data, num_para, data1, obs_1)

    # show the result
    plt.plot(data1, obs_1, 'r')

    plt.plot(data1, test_my_fun(a_est, b_est, data1), 'g')
    plt.show()


if __name__ == "__main__":
    test_on_LM()

































