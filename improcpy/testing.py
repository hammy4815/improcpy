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
        def test_flip(path="sample_images/cosmo.jpg", save=False):
            image = Testing.ColorTesting.before(path)
            title = "flipped color image"
            image.flip()
            Testing.after(image, title, save)

    class GrayTesting(object):
        @staticmethod
        def before(path):
            return GrayImage(path=path)

    class ImageProcessingTesting(object):
        @staticmethod
        def operation(*args, **kwargs):
            image = kwargs["image"]
            operation = kwargs["operation"]
            return operation(image, *args)


# image = Testing.ColorTesting.before(path="sample_images/cosmo.jpg")
# image = ip.flip(image)
# image.plot_image_full('flip',path='sample_tested/cosmo_flip')


image = imread("sample_images/cosmo.jpg")
ip.plot_image_full(image, "cosmo", path="sample_tested/cosmo")

image = ip.to_gray_scale(image)
ip.plot_image_full(image, "gray cosmo", path="sample_tested/cosmo_gray", color=False)

image = GrayImage(path="sample_images/cosmo.jpg")
X, Y = np.meshgrid(np.linspace(-10, 10, image.shape[1]), np.linspace(-10, 10, image.shape[0]))
H = np.exp(-(X ** 2 + Y ** 2) / (2.0 * 0.5 ** 2))
image = ip.dft_filter(image, H)
ip.plot_image_full(image, "blurry cosmo", path="sample_tested/cosmo_blend", color=False)


# from improcpy.image import ColorImage

# image = ColorImage(path="sample_images/worthen.jpg")
# image.plot_image_full(title="worthen", path="sample_tested/worthen.png")

# image.flip()
# image.plot_image_full(title="worthen flipped", path="sample_tested/worthen_flipped.png")

# new_image = image.cropper(250, 250, x=50, y=50, reset=True)
# new_image.plot_image_full(title="worthen partial", path="sample_tested/worthen_partial.png")

# image.plot_image_full(title="worthen unchanged", path="sample_tested/worthen_unchanged.png")

# image.hard_reset()
# image.plot_image_full(title="worthen restored", path="sample_tested/worthen_restored.png")
