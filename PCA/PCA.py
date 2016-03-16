import numpy as np


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