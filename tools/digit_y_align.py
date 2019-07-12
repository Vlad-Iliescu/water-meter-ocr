import cv2

from src import ImageProcessor
from src.config import config


class DigitYAlign:
    MAX = 300

    def __init__(self, image_path) -> None:
        self.min = config.getint(config.default_section, 'digit_min_height')
        self.max = config.getint(config.default_section, 'digit_max_height')

        self.t1 = config.getint(config.default_section, 'canny_threshold_1')
        self.t2 = config.getint(config.default_section, 'canny_threshold_2')

        self.y = config.getint(config.default_section, 'digit_y_alignment')

        self.window_name = "Size"
        self.processor = ImageProcessor(image_path)
        self.processor.fix_skew_angle()

    @staticmethod
    def get_boxes(image, boxes):
        if boxes is None:
            return image

        img = image.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img

    def process(self, value):
        self.y = value
        detected_edges = ImageProcessor.edges_detection(self.processor.base_image, self.t1, self.t2)
        contours = ImageProcessor.contours_detection(detected_edges)

        filtered, bounding_boxes = ImageProcessor.filter_contours(contours, self.min, self.max)
        aligned_boxes = ImageProcessor.max_aligned_boxes(bounding_boxes, self.y)

        image = self.get_boxes(self.processor.original_image, aligned_boxes)

        cv2.imshow(self.window_name, image)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Y:", self.window_name, self.y, DigitYAlign.MAX, lambda x: self.process(x))
        self.process(self.y)
        cv2.waitKey()


if __name__ == "__main__":
    canny = DigitYAlign('../wm.jpg')
    canny.run()
