import numpy as np
import matplotlib.pyplot as plt


def draw_curve(point_x, point_y, text, deg):

    x = np.linspace(0, 6, 100)
    plt.plot(point_x, point_y, 'bo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.text(5, 0.5, text)
    plt.title("Polynomial Curve Fitting")
    poly_para = np.polyfit(point_x, point_y, deg=deg)

    poly_func = np.poly1d(poly_para)
    plt.plot(x, poly_func(x), color="r", linewidth=2)
    plt.plot(x, np.sin(x), color="g", linewidth=2)

    plt.show()