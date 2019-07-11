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


def display_lines(img, lines, title):
    img = img.copy()

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    display(img, title)


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


def get_average_angle(lines):
    filtered_lines = []
    theta_min = 60 * np.pi / 180
    theta_max = 120 * np.pi / 180
    theta_avr = 0
    theta_deg = 0
    for line in lines:
        theta = line[0][1]
        if theta_min < theta < theta_max:
            filtered_lines.append(lines)
            theta_avr += theta

    if len(filtered_lines) > 0:
        theta_avr /= len(filtered_lines)
        theta_deg = (theta_avr / np.pi * 180) - 90

    return theta_deg


if __name__ == '__main__':
    d = False
    s = False
    orig = cv2.imread('counterraw.png')
    d and display(orig, "Original")

    # rotate 90 deg
    rotated = rotate(orig, 90)
    s and show_image(rotated, prev=orig)
    d and display(rotated, "Rotated 90")

    # grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    s and show_image(gray, prev=rotated)
    d and display(gray, "Greyscale")

    # edge detection
    edges = canny(gray, 200, 140)
    s and show_image(edges, prev=gray)
    d and display(edges, "Canny")

    # lines
    lines = hough_lines(edges)
    # show_image(lines, prev=edges)
    d and display_lines(rotated, lines, "Lines")

    # rotate again
    avg_angle = get_average_angle(lines)
    print(avg_angle)
    normal = rotate(gray, -avg_angle)
    s and show_image(normal, prev=gray)
    d and display(normal, "Normal")

    cv2.waitKey()
