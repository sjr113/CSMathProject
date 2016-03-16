import numpy as np


def add_Gaussian_noise_on_points(point, num_point):
    # point: given point_x and point_y
    # para: the intensity of noise

    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, num_point)

    return noise + point