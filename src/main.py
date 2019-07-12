from src import ImageProcessor

if __name__ == '__main__':
    processor = ImageProcessor('../wm.jpg')
    processor._debug = True
    processor.process()
