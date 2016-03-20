import numpy as np
import load_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import NullLocator


def principle_component_analysis(data_set, top_nfeature=9999999):

    # compute the mean of the data set
    mean_val = np.mean(data_set, axis=0)
    # minus the mean of the data set to make the data set more soft
    mean_removed = data_set - mean_val

    # compute the convariance matrix
    cov_mat = np.cov(mean_removed, rowvar=0)
    # compute the eigen_value and eigen_vector of the convariance matrix
    eig_val, eig_vector = np.linalg.eig(np.mat(cov_mat))

    # sort the eigen_value of that convariance matrix
    eig_val_ind = np.argsort(eig_val)
    # get the given number (top_nfeature) eigen_values of the convariance matrix
    eig_val_sort = eig_val_ind[:-(top_nfeature + 1): -1]
    # get the given number (top_nfeature) eigen_vectors of the convariance matrix
    red_eig_vector = eig_vector[:, eig_val_sort]

    # transfer the original data set to the new feature space to get the reconstructed data set
    low_data_set = mean_removed * red_eig_vector
    recon_data = (low_data_set * red_eig_vector.transpose()) + mean_val

    return low_data_set, recon_data


def find_typical_points(low_data_set):
    max_value = 30
    min_value = -20

    x = np.linspace(start=min_value, stop=max_value, num=5, endpoint=False)

    # next we find some key points
    arg_index = []
    # we need to visualize 25 images
    point_location = []
    for i in range(5):
        for j in range(5):
            point_location.append((x[i], x[j]))   # generate 25 points

    # 2. find the key points
    for i in range(len(point_location)):
        point = point_location[i]
        shortest_distance = 99999999
        arg = 0
        for j in range(np.shape(low_data_set)[0]):
            distance = dist_points(point[0], point[1], low_data_set[j, 0], low_data_set[j, 1])
            if distance < shortest_distance:
                shortest_distance = distance
                arg = j
        arg_index.append(arg)

    return arg_index


def dist_points(x1, y1, x2, y2):
    return ((x1-x2)**2 + (y1-y2)**2) ** 0.5


def test_on_PCA():

    data_without_label, data_set = load_dataset.data_processing()

    low_data_set, recon_data = principle_component_analysis(data_without_label, top_nfeature=2)

    plt.plot(low_data_set[:, 0], low_data_set[:, 1], 'bo')
    plt.xlabel("x")
    plt.ylabel("y")
    text = "PCA"
    plt.text(4.5, 0.5, text)
    plt.title("Principle Component Analysis")
    plt.grid(True)   # add grid to the picture

    # draw the images of these key points
    arg_index = find_typical_points(low_data_set)

    # major_locator = NullLocator()
    # plot the points nearest these points
    for i in range(25):
        plt.plot(low_data_set[arg_index[i], 0], low_data_set[arg_index[i], 1], 'ro')
    # to make the ticks not show
    # fig, ax = plt.subplots()
    # plt.plot(...)
    # ax.xaxis.set_major_locator(major_locator)
    plt.show()

    # # draw the picture of 3
    # index = 1  # just draw the reconstruction image of red point
    # image = recon_data[index, :].reshape((8, 8))
    # plt.imshow(image, cmap=cm.gray_r)
    # plt.show()

    k = 0
    for i in range(5):
        for j in range(5):
            k += 1
            ax = plt.subplot(5, 5, k)
            # to make the ticks not show
            ax.xaxis.set_major_locator((NullLocator()))
            ax.yaxis.set_major_locator((NullLocator()))
            image = recon_data[arg_index[5 * i + j], :].reshape((8, 8))
            plt.imshow(image, cmap=cm.gray_r)
    plt.show()

if __name__ == "__main__":
    test_on_PCA()

    # The code to test the ticks
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    #
    # ax.xaxis.set_major_locator((NullLocator()))
    # ax.yaxis.set_major_locator((NullLocator()))
    #
    # plt.plot([1, 1], [2, 3])
    #
    # plt.show()
