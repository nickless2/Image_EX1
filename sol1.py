from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2grey
import sys

#
def input_validation(fileame, representation):

    # check if file path is valid
    try:
        im = imread(fileame)
    except:
        print('%s is not a valid image file, please check the path and try '
              'again' %fileame)
        sys.exit(1)

    # check if representation # is valid
    if representation != 1 and representation != 2:
        print('%s is an invalid representation number, please try again'
              %representation)
        sys.exit(1)

    return im


def read_image(fileame, representation):

    im = input_validation(fileame, representation)
    #check if we want to convert RGB pic to greyscale
    if representation == 1 and im.shape.__len__() == 3:
        im = rgb2grey(im)
        im = im.astype(np.float32)

    else:
        im = im.astype(np.float32)
        im /= 255
        print(2)

    return im

read_image('23.jpg', 1)