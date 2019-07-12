import cv2
import numpy as np

from src.config import config


class ImageProcessor:
    def __init__(self, image_path) -> None:
        self._image_path = image_path
        self._debug = config.getboolean(config.default_section, 'debug')

        self.original_image = cv2.imread(self._image_path)
        self.show_image(self.original_image, "1. Original Image")

        self.original_image = self.rotate(self.original_image, config.getint(config.default_section, 'rotation_deg'))
        self.show_image(self.original_image, "2. Original Image Rotated")

        self.base_image = self.grey_out(self.original_image)
        self.show_image(self.base_image, "3. Gray image")

        self.rois = []

    def show_image(self, image, title):
        if self._debug:
            cv2.imshow(title, image)
            cv2.waitKey(1)

    def show_lines(self, image, lines, title):
        if self._debug:
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

            self.show_image(img, title)

    def show_contours(self, image, contours, title):
        if self._debug:
            img = image.copy()
            cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
            self.show_image(img, title)

    def show_boxes(self, image, boxes, title):
        if self._debug:
            img = image.copy()
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.show_image(img, title)

    def show_rois(self, rois, title):
        if self._debug:
            rois = rois.copy()
            h_min = min(im.shape[0] for im in rois)
            rois = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
                    for im in rois]
            img = rois[0]
            for roi in rois[1:]:
                img = np.concatenate((img, roi), axis=1)

            self.show_image(img, title)

    def fix_skew_angle(self):
        # edge detection
        edges = self.edges_detection(
            self.base_image,
            config.getint(config.default_section, 'canny_threshold_1'),
            config.getint(config.default_section, 'canny_threshold_2')
        )
        self.show_image(edges, "4. Edges")

        # line detection
        lines = self.lines_detection(edges, config.getint(config.default_section, 'line_threshold'))
        self.show_lines(self.original_image, lines, "5. Lines")

        # detect angle
        avg_angle = self.average_angle(lines)

        # skew rotate
        self.original_image = self.rotate(self.original_image, -avg_angle)
        self.base_image = self.grey_out(self.original_image)
        self.show_image(self.original_image, "6. Skew fix")

    def digits_bounding_boxes(self, edges):
        # contours detection
        contours = self.contours_detection(edges)
        self.show_contours(self.original_image, contours, "8. All contours")
        filtered_contours, bounding_boxes = self.filter_contours(
            contours,
            config.getint(config.default_section, 'digit_min_height'),
            config.getint(config.default_section, 'digit_max_height')
        )
        self.show_contours(self.original_image, filtered_contours, "9. Filtered contours")
        self.show_boxes(self.original_image, bounding_boxes, "10. Filtered contours boxes")

        # find aligned boxes
        aligned_boxes = self.max_aligned_boxes(bounding_boxes,
                                               config.getint(config.default_section, 'digit_y_alignment'))
        return self.sorted_boxes(aligned_boxes)

    def process(self):
        self.fix_skew_angle()

        # edge detection
        edges = self.edges_detection(
            self.base_image,
            config.getint(config.default_section, 'canny_threshold_1'),
            config.getint(config.default_section, 'canny_threshold_2')
        )
        self.show_image(edges, "7. Normal Edges")

        sorted_boxes = self.digits_bounding_boxes(edges)
        self.show_boxes(self.original_image, sorted_boxes, "11. Aligned boxes")

        self.rois = self.rois_cut(edges, sorted_boxes)
        self.show_rois(self.rois, "12. ROIs")

        if self._debug:
            cv2.waitKey()

    @staticmethod
    def grey_out(original_image):
        return cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def rotate(original_image, degrees=90):
        if degrees == 0:
            return original_image

        height, width = original_image.shape[:2]
        center_x, center_y = (width / 2, height / 2)

        M = cv2.getRotationMatrix2D((center_x, center_y), -degrees, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        M[0, 2] += (new_width / 2) - center_x
        M[1, 2] += (new_height / 2) - center_y

        return cv2.warpAffine(original_image, M, (new_width, new_height))

    @staticmethod
    def edges_detection(original_image, threshold1, threshold2):
        return cv2.Canny(original_image, threshold1, threshold2)

    @staticmethod
    def lines_detection(original_image, threshold):
        return cv2.HoughLines(original_image, 1, np.pi / 180, threshold)

    @staticmethod
    def average_angle(lines):
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

    @staticmethod
    def contours_detection(original_image):
        contours, _ = cv2.findContours(original_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def filter_contours(contours, min_height, max_height):
        bounding_boxes = []
        filtered_contours = []
        for contour in contours:
            bounds = cv2.boundingRect(contour)
            width, height = bounds[2:4]
            if height > min_height and max_height > height > width:
                bounding_boxes.append(bounds)
                filtered_contours.append(contour)

        return filtered_contours, bounding_boxes

    @staticmethod
    def aligned_boxes(boxes, threshold):
        start = boxes[0]
        result = [start]
        for box in boxes[1:]:
            x, y, w, h = box
            if abs(start[1] - y) < threshold and abs(start[3] - h) < 5:
                result.append(box)

        return result

    @staticmethod
    def max_aligned_boxes(bounding_boxes, threshold):
        aligned = []
        for i, box in enumerate(bounding_boxes):
            tmp = ImageProcessor.aligned_boxes(bounding_boxes[i:], threshold)
            if len(tmp) > len(aligned):
                aligned = tmp

        return aligned

    @staticmethod
    def sorted_boxes(bounding_boxes):
        boxes = bounding_boxes.copy()
        boxes.sort(key=lambda x: x[0])  # sort by x
        return boxes

    @staticmethod
    def rois_cut(edges, bounding_boxes):
        rois = []
        for box in bounding_boxes:
            x, y, w, h = box
            roi = edges[y:y + h, x:x + w]
            rois.append(roi)

        return rois

    @staticmethod
    def black_and_white(original_image):
        (_, black_white) = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

        return black_white

    @staticmethod
    def blur(original_image):
        blur = cv2.blur(original_image, (5, 5))

        return blur
