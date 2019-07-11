import cv2
import numpy as np

window_name = "lines"
lowThreshold = 1
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


def img_lines(img, lines):
    img = orig.copy()
    if lines is None:
        return img

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

    return img


def do_lines(val):
    global lowThreshold
    lowThreshold = val or 1
    print(lowThreshold)
    lines = cv2.HoughLines(gray, 1, np.pi / 180, lowThreshold)

    cv2.imshow(window_name, img_lines(gray, lines))


if __name__ == '__main__':
    orig = cv2.imread('counterraw.png')
    orig = rotate(orig, 90)
    lowThreshold = 1

    # grayscale
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 200, 140)


    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar("Low Threshold:", window_name, lowThreshold, 500, do_lines)

    do_lines(0)

    cv2.waitKey()
