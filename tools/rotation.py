import cv2

from src import ImageProcessor
from src.config import config


class Rotation:
    MAX = 360

    def __init__(self, image_path) -> None:
        self.degree = config.getint(config.default_section, 'rotation_deg')
        self.window_name = "Rotation"
        self.processor = ImageProcessor(image_path)

    def process(self, value):
        self.degree = value
        print(self.degree)
        rotated = ImageProcessor.rotate(self.processor.original_image, self.degree)
        cv2.imshow(self.window_name, rotated)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Deg:", self.window_name, self.degree, Rotation.MAX, lambda x: self.process(x))
        self.process(self.degree)
        cv2.waitKey()


if __name__ == "__main__":
    canny = Rotation('../apometru.jpg')
    canny.run()
