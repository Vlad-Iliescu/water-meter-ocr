from src import ImageProcessor
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

if __name__ == '__main__':
    processor = ImageProcessor('wm.jpg')
    # processor._debug = True
    processor.process()
    image = processor.rois_to_image()
    image = ImageProcessor.black_and_white(image)
    img = Image.fromarray(image)
    img.save('rois.png')
    string = pytesseract.image_to_string(img)
    print(string)

    print("Digits Found = {}".format(len(processor.rois)))
    print("OCR = '{}'".format(string))
