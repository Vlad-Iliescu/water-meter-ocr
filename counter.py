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


def filter_contours(contours):
    bounding_boxes = []
    filtered_contours = []
    min_height = 20
    max_height = 90
    for contour in contours:
        bounds = cv2.boundingRect(contour)
        height = bounds[3]
        width = bounds[2]
        if height > min_height and max_height > height > width:
            bounding_boxes.append(bounds)
            filtered_contours.append(contour)

    return bounding_boxes, filtered_contours


def find_aligned(boxes):
    start = boxes[0]
    result = [start]
    for box in boxes[1:]:
        if abs(start[1] - box[1]) < 10 and abs(start[3] - box[3]) < 5:
            result.append(box)

    return result


def find_all_aligned(bounding_boxes):
    aligned = []
    for i, box in enumerate(bounding_boxes):
        tmp = find_aligned(bounding_boxes[i:])
        if len(tmp) > len(aligned):
            aligned = tmp

    return aligned


def display_contours(img, contours, title):
    img = img.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
    display(img, title)


def display_boxes(img, boxes, title):
    img = img.copy()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    display(img, title)


def sort_by_x(boxes):
    boxes = boxes.copy()
    boxes.sort(key=lambda x: x[0])
    return boxes


def get_rois(img, boxes):
    rois = []
    for box in boxes:
        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        rois.append(roi)

    return rois


if __name__ == '__main__':
    d = False
    s = False
    orig = cv2.imread('counterraw.png')
    d and display(orig, "1. Original")

    # rotate 90 deg
    rotated = rotate(orig, 90)
    s and show_image(rotated, prev=orig)
    d and display(rotated, "2. Rotated 90")

    # grayscale
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    s and show_image(gray, prev=rotated)
    d and display(gray, "3. Greyscale")

    # edge detection
    edges = canny(gray, 120, 180)
    s and show_image(edges, prev=gray)
    d and display(edges, "4. Canny")

    # lines
    lines = hough_lines(edges)
    # show_image(lines, prev=edges)
    d and display_lines(rotated, lines, "5. Lines")

    # rotate again for skew angle
    avg_angle = get_average_angle(lines)
    print(avg_angle)
    normal = rotate(gray, -avg_angle)
    s and show_image(normal, prev=gray)
    d and display(normal, "6. Normal")

    # noraml edge detection
    normal_edges = canny(normal, 120, 280)
    s and show_image(normal_edges, prev=normal)
    d and display(normal_edges, "7. Normal Canny")

    # contours
    contours, hierarchy = cv2.findContours(normal_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes, filtered_contours = filter_contours(contours)
    skew_normal = rotate(rotated, -avg_angle)
    d and display_contours(skew_normal, filtered_contours, "8. Normal Contours")
    d and display_boxes(skew_normal, bounding_boxes, "9. Normal Boxes")

    # aligned boxes
    aligned_boxes = find_all_aligned(bounding_boxes)
    d and display_boxes(skew_normal, aligned_boxes, "10. Aligned Boxes")
    aligned_boxes_by_x = sort_by_x(aligned_boxes)

    rois = get_rois(skew_normal, aligned_boxes_by_x)
    cv2.imshow('segment no:', rois[1])
    print(rois)

    cv2.waitKey()
