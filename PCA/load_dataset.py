import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def load_dataset(file_name):
    # This method wastes much memory.
    # file = open("optdigits.tes")
    file = open(file_name)
    result = []

    while 1:
        line = file.readline()

        if not line:
            break
        # note that the final string is "\n", not the label of the image
        if line[len(line)-2] == "3":
            result.append(map(int, line.split(",")))
    file.close()

    # cPickle.dump(result, open("data_tes", "wb"))
    cPickle.dump(result, open("data_" + file_name, "wb"))


def data_processing():

    load_dataset(file_name="optdigits.tra")
    load_dataset(file_name="optdigits.tes")

    # read data from the data file
    data_tra = cPickle.load(open("data_optdigits.tra", "rb"))
    data_tes = cPickle.load(open("data_optdigits.tes", "rb"))
    data = np.vstack((data_tra, data_tes))

    data_without_label = np.zeros((len(data), 64))
    data_without_label = data[:, :-1]
    # To save the original data set, we must use the "np.copy", that is called deep copy.
    # if you use the "a = b", then it is only copy the point, which means shallow copy.
    data_without_label_origin = np.copy(data_without_label)
    # print np.shape(data_without_label)

    # Next we reshape all images to (8*8)
    all_image_reshape = np.zeros((np.shape(data_without_label)[0], 8, 8))
    # print data_without_label[1, :]
    for i in range(np.shape(data_without_label)[0]):
        data_without_label[i, np.argwhere(data_without_label[i, :] != 0)] = 1
        all_image_reshape[i, :, :] = data_without_label[i, :].reshape((8, 8))

    # print all_image_reshape[1, :, :]
    return data_without_label_origin, all_image_reshape


if __name__ == "__main__":

    data_without_label, all_image_reshape = data_processing()

    # Just test the image
    image = data_without_label[3, :].reshape((8, 8))

    plt.imshow(image, cmap=cm.gray_r)
    plt.show()

    for i in range(8):
        for j in range(8):
            if image[i][j] > 0:
                image[i][j] = 1
    # print image

    # use the matplotlib method show the image
    # note here I add the parameter "cmap=cm.gray_r" to make it show the gray image better
    plt.imshow(image, cmap=cm.gray_r)
    plt.show()

    # # use the opencv method to show the images
    # # win = cv2.namedWindow('test win', flags=0)
    # # cv2.imshow('test win', image)
    # # cv2.waitKey(0)




















