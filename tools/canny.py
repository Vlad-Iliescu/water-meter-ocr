import cv2

from src import ImageProcessor
from src.config import config


class Canny:
    MAX = 500

    def __init__(self, image_path) -> None:
        self.low = config.getint(config.default_section, 'canny_threshold_1')
        self.high = config.getint(config.default_section, 'canny_threshold_2')
        self.window_name = "Canny"
        self.processor = ImageProcessor(image_path)

    def process(self):
        print(self.low, self.high)
        detected_edges = ImageProcessor.edges_detection(self.processor.base_image, self.low, self.high)
        cv2.imshow(self.window_name, detected_edges)

    def process_low(self, value):
        self.low = value
        self.process()

    def process_high(self, value):
        self.high = value
        self.process()

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Low:", self.window_name, self.low, Canny.MAX, lambda x: self.process_low(x))
        cv2.createTrackbar("High:", self.window_name, self.high, Canny.MAX, lambda x: self.process_high(x))
        self.process()
        cv2.waitKey()


if __name__ == "__main__":
    canny = Canny('../wm.jpg')
    canny.run()
