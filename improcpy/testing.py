from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
from improcpy.image import ColorImage, GrayImage
from improcpy.image import ImageProcessing as ip


class Testing(object):
    @staticmethod
    def after(image, title, save):
        plt.figure()
        image.plot_image(title)
        if not save:
            plt.show()
        else:
            plt.savefig(title)

    class ColorTesting(object):
        @staticmethod
        def before(path):
            return ColorImage(path=path)

        @staticmethod
        def test_flip(path="sample_images/man.jpg", save=False):
            image = Testing.ColorTesting.before(path)
            title = "flipped color image"
            image.flip()
            Testing.after(image, title, save)

    class GrayTesting(object):
        @staticmethod
        def before(path):
            return GrayImage(path=path)

        @staticmethod
        def test_flip(path="sample_images/man.jpg", save=False):
            image = Testing.GrayTesting.before(path)
            title = "flipped grey image"
            image.flip()
            Testing.after(image, title, save)


# Testing.GrayTesting.test_flip()
# Testing.ColorTesting.test_flip()

image = Testing.ColorTesting.before(path="sample_images/man.jpg")
image = ip.flip(image)
image.plot_image_full()

