import cv2
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray
import numpy as np


def show_image(img, title="Image", prev=None):
    if prev is not None:
        plt.subplot(121), plt.imshow(prev, cmap='gray')
        plt.title("Original"), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()


def display(img, title="Image"):
    cv2.imshow(title, img)


def rotate(img: ndarray, degrees: int):
    height, width = img.shape[:2]
    center_x, center_y = (width / 2, height / 2)

    M = cv2.getRotationMatrix2D((center_x, center_y), -degrees, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    M[0, 2] += (new_width / 2) - center_x
    M[1, 2] += (new_height / 2) - center_y

    rotated = cv2.warpAffine(img, M, (new_width, new_height))

    return rotated


def black_and_white(img):
    (thresh, black_white) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return black_white


def blur(img):
    blur = cv2.blur(img, (5, 5))

    return blur


def canny(img, threshold1=200, threshold2=240):
    edges = cv2.Canny(img, threshold1, threshold2)

    return edges


def hough_lines(img):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 140)

    return lines


if __name__ == '__main__':
    d = False
    s = False
    orig = cv2.imread('counterraw.png')
    d and display(orig, "Original")

    # grayscale
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    s and show_image(gray, prev=orig)
    d and display(gray, "Greyscale")

    # rotate 90 deg
    rotated = rotate(gray, 90)
    s and show_image(rotated, prev=gray)
    d and display(rotated, "Rotated 90")

    # edge detection
    edges = canny(rotated, 200, 240)
    s and show_image(edges, prev=rotated)
    d and display(edges, "Canny")

    # lines

    cv2.waitKey()
