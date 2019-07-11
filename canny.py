import cv2
import numpy as np

window_name = "canny"
lowThreshold = 10
highThreshold = 110
gray = None

def rotate(img, degrees):
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

def do_canny():
    print(lowThreshold, highThreshold)
    detected_edges = cv2.Canny(gray, lowThreshold, highThreshold, 4)
    cv2.imshow(window_name, detected_edges)


def do_canny_low(val):
    global lowThreshold
    lowThreshold = val
    do_canny()


def do_canny_high(val):
    global highThreshold
    highThreshold = val
    do_canny()


if __name__ == '__main__':
    orig = cv2.imread('counterraw.png')
    lowThreshold = 100
    highThreshold = 200

    # grayscale
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray = rotate(gray, 90)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("Low Threshold:", window_name, lowThreshold, 500, do_canny_low)
    cv2.createTrackbar("High Threshold:", window_name, highThreshold, 500, do_canny_high)

    do_canny_low(0)

    cv2.waitKey()
