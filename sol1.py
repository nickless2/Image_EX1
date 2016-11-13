from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2grey
import sys


def input_validation(filename, representation):

    """
    makes sure that 'filename' is a correct path to an image file.
    makes sure that 'representation' is 1 or 2
    """
    # check if file path is valid
    try:
        im = imread(filename)
    except:
        print('%s is not a valid image file, please check the path and try '
              'again' %filename)
        sys.exit(1)

    # check if representation # is valid
    if representation != 1 and representation != 2:
        print('%s is an invalid representation number, please try again'
              %representation)
        sys.exit(1)

    return im


def read_image(filname, representation):
    """
    reads a given image and changes its representation if necessary
    """

    im = input_validation(filname, representation)
    #check if we want to convert RGB pic to greyscale
    if representation == 1 and im.shape.__len__() == 3:
        im = rgb2grey(im)
        im = im.astype(np.float32)

    else:
        im = im.astype(np.float32)
        im /= 255

    return im


def imdisplay(filename, representation):

    """
    displays an image with a given representation, uses 'read_image'
    """
    im = read_image(filename, representation)
    #check representation type
    representationType = plt.cm.gray if representation == 1 else None
    plt.imshow(im, cmap=representationType)
    plt.show()


def rgb2yiq(imRGB):
    """
    transforms a rgb picture to yiq
    """
    matrix_2_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],
                             [0.212, -0.523, 0.311]])

    height = imRGB.shape[0]
    width = imRGB.shape[1]
    yiq_pic = np.zeros([height, width, 3])

    for i in range(0, 3):
        yiq_pic[:, :, 0] += matrix_2_YIQ[0, i] * imRGB[:, :, i]
        yiq_pic[:, :, 1] += matrix_2_YIQ[1, i] * imRGB[:, :, i]
        yiq_pic[:, :, 2] += matrix_2_YIQ[2, i] * imRGB[:, :, i]

    return yiq_pic


def yiq2rgb(imRGB):
    """
    transforms a yiq picture to rgb
    """

    matrix_2_rgb = np.array([[1, 0.956, 0.621], [1, -0.272, -0.647],
                             [1, -1.106, 1.703]])

    height = imRGB.shape[0]
    width = imRGB.shape[1]
    RGB_pic = np.zeros([height, width, 3])

    for i in range(0, 3):
        RGB_pic[:, :, 0] += matrix_2_rgb[0, i] * imRGB[:, :, i]
        RGB_pic[:, :, 1] += matrix_2_rgb[1, i] * imRGB[:, :, i]
        RGB_pic[:, :, 2] += matrix_2_rgb[2, i] * imRGB[:, :, i]

    return RGB_pic

im = read_image('jerusalem.jpg', 2)
im_yiq=rgb2yiq(im)
yiq2rgb(im_yiq)