from scipy.misc import imread as imread, imsave as imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
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


def read_image(filename, representation):
    """
    reads a given image and changes its representation if necessary
    """

    im = input_validation(filename, representation)
    #check if we want to convert RGB pic to greyscale
    if representation == 1 and im.shape.__len__() == 3:
        im = rgb2gray(im)
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
    representation_type = plt.cm.gray if representation == 1 else None
    plt.imshow(im, cmap=representation_type)
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


#todo add input checks
def histogram_equalize(im_orig):

    is_RGB = False

    #if image is a RGB
    if im_orig.shape.__len__() == 3:
        is_RGB = True
        yiq_im = rgb2yiq(im_orig)
        y_im = yiq_im[:, :, 0]
        image = y_im
    else:
        image = im_orig

    # transfer back to uint8 representation
    image *= 255
    image = np.round(image)
    image = image.astype(np.uint8)


    hist, bin_edges = np.histogram(image, 256, [0, 256])

    # cumulative distribution function
    cdf = np.cumsum(hist)
    # todo
    #cdf = cdf / image.size * 255

    min_val_cdf = np.min(cdf[np.nonzero(cdf)])
    cdf = np.round(((cdf - min_val_cdf) / (image.size - min_val_cdf)) * 255)

    # perform linear interpolation to get equalized image
    eq_image = np.interp(image, bin_edges[:-1], cdf)

    eq_image_hist, eq_image_bin_edges = np.histogram(eq_image, 256, [0, 256])

    # convert bak to float32 representation
    eq_image = eq_image.astype(np.float32)
    eq_image /= 255

    if is_RGB:
        yiq_im[:, :, 0] = eq_image
        eq_image = np.clip(yiq2rgb(yiq_im), 0, 1)

    return [eq_image, hist, eq_image_hist]


def quantize (im_orig, n_quant, n_iter):

    # todo input check

    # transfer values to [0, 255] format
    image = im_orig * 255
    image = np.round(image)
    image = image.astype(np.uint8)

    # get histogram and cumsum
    hist = np.histogram(image, 256, [0, 256])[0]
    cdf = np.cumsum(hist)

    # calculate initial z values
    z_arr = np.array([0, 255])
    ppi = np.round(im_orig.size / n_quant)  # ppi = pixel per interval
    for i in range(1, n_quant):
        z_index = np.argwhere(cdf > i * ppi)[0, 0]
        z_arr = np.insert(z_arr, i, z_index)
    print(z_arr)
    # calculate q values
    q_arr = np.zeros(n_quant)
    for i in range(n_iter):
        for j in range(n_quant):
            zxc = range(z_arr[j], z_arr[j+1])
            print(zxc)
            q_arr[j] = np.average(zxc, weights=hist[zxc])
            print(q_arr)

            # nominator = 0
            # denominator = 0
            # for k in range(z_arr[j], z_arr[j+1] +1):
            #     nominator += k * hist[k]
            #     denominator += hist[k]
            # q_arr[j] = nominator / denominator

        #calculate neq z's
        # for k in range(1, z_arr.size - 1):
        #     z_arr[i] = q_arr[i-1] + q_arr[i]







if __name__ == '__main__':


    x = read_image('LowContrast.jpg', 2)
    quantize(x, 4, 4)

    #print(np.averge(data))
    #histogram_equalize(x)


