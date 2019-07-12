import cv2
import numpy as np
from src import ImageProcessor
from src.config import config


class Lines:
    MAX = 200

    def __init__(self, image_path) -> None:
        self.threshold = config.getint(config.default_section, 'line_threshold')
        self.low = config.getint(config.default_section, 'canny_threshold_1')
        self.high = config.getint(config.default_section, 'canny_threshold_2')
        self.window_name = "Lines"
        self.processor = ImageProcessor(image_path)

    @staticmethod
    def get_line_image(image, lines):
        if lines is None:
            return image

        img = image.copy()

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

    def process(self, value):
        self.threshold = value
        print(self.threshold)
        detected_edges = ImageProcessor.edges_detection(self.processor.base_image, self.low, self.high)
        lines = ImageProcessor.lines_detection(detected_edges, self.threshold)

        with_lines = self.get_line_image(self.processor.original_image, lines)
        cv2.imshow(self.window_name, with_lines)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Threshold:", self.window_name, self.threshold, Lines.MAX, lambda x: self.process(x))
        self.process(self.threshold)
        cv2.waitKey()


if __name__ == "__main__":
    canny = Lines('../wm.jpg')
    canny.run()
