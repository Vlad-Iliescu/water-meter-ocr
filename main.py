from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt


def plot(img, edges):
    plt.subplot(121), plt.imshow(img.copy(), cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def show_lines(img, lines):
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)


def get_average_angle(lines):
    filtered_lines = []
    theta_min = 60 * np.pi / 180
    theta_max = 120 * np.pi / 180
    theta_avr = 0
    theta_deg = 0
    for i in range(len(lines)):
        theta = lines[i][0][1]
        if theta_min < theta < theta_max:
            filtered_lines.append(lines[i])
            theta_avr += theta

    if len(filtered_lines) > 0:
        theta_avr /= len(filtered_lines)
        theta_deg = (theta_avr / np.pi * 180) - 90

    return theta_deg


def plot_contours(img, contours):
    img = img.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 255), 3)
    cv2.imshow("title", img)
    cv2.waitKey()


def filter_contours(contours):
    bounding_boxes = []
    filtered_contours = []
    min_height = 0
    max_height = 500
    for i in range(len(contours)):
        bounds = cv2.boundingRect(contours[i])
        height = bounds[2]
        width = bounds[3]
        if height > min_height and max_height > height > width:
            bounding_boxes.append(bounds)
            filtered_contours.append(contours[i])

    return bounding_boxes, filtered_contours


if __name__ == '__main__':
    ref = cv2.imread('counterraw.png')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # ref = ~ref
    # ref = cv2.bitwise_not(ref)

    edges = cv2.Canny(ref, 40, 250)
    # plot(ref, edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)
    # show_lines(ref, lines)
    avg_angle = get_average_angle(lines)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes, filtered_contours = filter_contours(contours)

    plot_contours(ref, contours)
    plot_contours(ref, filtered_contours)

    # ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('image', ref)
    # cv2.waitKey(0)
