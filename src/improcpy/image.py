from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from exceptions import ImproperImageFormatException
from PIL import Image

# This decorator is used to prevent or allow changes to Image upon transformations
def reset_image(func):
    def wrapper(*args, **kwargs):
        returned_image = func(*args, **kwargs)
        if kwargs.get("reset", None) or args[-1] == True:
            args[0].soft_reset()
        else:
            args[0].temp = returned_image
        return returned_image

    return wrapper


# This decorator is used to allow numpy arrays or Image objects inside of ImageProcessing
def correct_image(func):
    def wrapper(*args, **kwargs):
        if isinstance(kwargs.get("image", None), Image):
            image_instance = kwargs.get("image", None)
            kwargs["image"] = image_instance.image
            image_instance.update_image(func(*args, **kwargs))
            return image_instance
        if len(args) > 0 and isinstance(args[0], Image):
            args = [*args]
            image_instance = args[0]
            args[0] = image_instance.image
            image_instance.update_image(func(*args, **kwargs))
            return image_instance
        else:
            return func(*args, **kwargs)

    return wrapper


# This decorator is used to allow numpy arrays or Image objects inside of ImageProcessing for two image inputs
def correct_images(func):
    def wrapper(*args, **kwargs):
        # if isinstance(kwargs.get("image", None), Image):
        #     image_instance = kwargs.get("image", None)
        #     kwargs["image"] = image_instance.image
        #     image_instance.update_image(func(*args, **kwargs))
        #     return image_instance
        # if len(args) > 0 and isinstance(args[0], Image):
        #     args = [*args]
        #     image_instance = args[0]
        #     args[0] = image_instance.image
        #     image_instance.update_image(func(*args, **kwargs))
        #     return image_instance
        # else:
        return func(*args, **kwargs)

    return wrapper


class Image(object):
    def __init__(self, image):
        """
            image [ndarray]: 
        """
        self.image = np.array(image)
        self.temp = np.array(image)
        self.default = np.array(image)

    def __str__(self):
        return str(self.image)

    def update_image(self, image):
        if isinstance(image, Image):
            self.image = image.image
        else:
            self.image = np.array(image)
        self.temp = np.array(image)
        self.default = np.array(image)

    def hard_reset(self):
        """Resets the image back to its very beginning"""
        self.image = self.default

    def soft_reset(self):
        """Resets the image back one step and can only be performed within a method"""
        self.image = self.temp

    def plot_image(self, title=""):
        ImageProcessing.plot_image(self.image, title, color=isinstance(self, ColorImage))

    def plot_image_full(self, title=""):
        ImageProcessing.plot_image_full(self.image, title, color=isinstance(self, ColorImage))

    @reset_image
    def flip(self, reset=False):
        """See ImageProcessing.flip"""
        self.image = ImageProcessing.flip(self.image)
        return self.image

    @reset_image
    def red_channel(self, reset=False):
        """See ImageProcessing.red_channel"""
        self.image = ImageProcessing.red_channel(self.image)
        return self.image

    @reset_image
    def green_channel(self, reset=False):
        """See ImageProcessing.green_channel"""
        self.image = ImageProcessing.green_channel(self.image)
        return self.image

    @reset_image
    def blue_channel(self, reset=False):
        """See ImageProcessing.blue_channel"""
        self.image = ImageProcessing.blue_channel(self.image)
        return self.image

    @reset_image
    def color_channel(self, color="r", reset=False):
        """See ImageProcessing.color_channel"""
        self.image = ImageProcessing.color_channel(self.image, color)
        return self.image

    @reset_image
    def thresholder(self, c, reset=False):
        """See ImageProcessing.thresholder"""
        self.image = ImageProcessing.thresholder(self.image, c)
        return self.image

    @reset_image
    def cropper(self, width, height, x=0, y=0, reset=False):
        """See ImageProcessing.cropper"""
        self.image = ImageProcessing.cropper(self.image, width, height, x, y)
        return self.image

    @reset_image
    def scaler(self, reset=False):
        """See ImageProcessing.scaler"""
        self.image = ImageProcessing.scaler(self.image)
        return self.image

    @reset_image
    def light_field(h, w, reset=False):
        """See ImageProcessing.light_field"""
        self.image = ImageProcessing.light_field(self.image, h, w)
        return self.image

    @reset_image
    def contrast_adjust(self, c, reset=False):
        """See ImageProcessing.contrast_adjust"""
        self.image = ImageProcessing.contrast_adjust(self.image, c, color=isinstance(self, ColorImage))
        return self.image

    @reset_image
    def bright_adjust(self, c, reset=False):
        """See ImageProcessing.bright_adjust"""
        self.image = ImageProcessing.bright_adjust(self.image, c, color=isinstance(self, ColorImage))
        return self.image

    @reset_image
    def alpha_blend(self, image2, alpha=0.5):
        """See ImageProcessing.alpha_blend"""
        self.image = ImageProcessing.alpha_blend(self.image, image2, alpha)
        return self.image

    @reset_image
    def cross_dissolve(self, image2, num_steps=10):
        """See ImageProcessing.cross_dissolve"""
        self.image = ImageProcessing.cross_dissolve(self.image, image2, num_steps)
        return self.image

    @reset_image
    def blur_gray(self, size=3):
        """See ImageProcessing.blur_gray"""
        self.image = ImageProcessing.blur_gray(self.image, size)
        return self.image

    @reset_image
    def median_filter_gray(self, size=3):
        """See ImageProcessing.median_filter_gray"""
        self.image = ImageProcessing.median_filter_gray(self.image, size)
        return self.image

    @reset_image
    def convolution_gray(self, kernel):
        """See ImageProcessing.convolution_gray"""
        self.image = ImageProcessing.convolution_gray(self.image, kernel)
        return self.image

    @reset_image
    def sharpen_gray(self):
        """See ImageProcessing.sharpen_gray"""
        self.image = ImageProcessing.sharpen_gray(self.image)
        return self.image

    @reset_image
    def edge_detect_gray(self):
        """See ImageProcessing.edge_detect_gray"""
        self.image = ImageProcessing.edge_detect_gray(self.image)
        return self.image

    @reset_image
    def frame(self, frame, translate=[0, 0], rotation=0, scale=1):
        """See ImageProcessing.frame"""
        self.image = ImageProcessing.frame(self.image, frame, translate=[0, 0], rotation=0, scale=1)
        return self.image
    
    @reset_image
    def dft_filter(image, filter):
        """See ImageProcessing.dft_filter"""
        self.image = ImageProcessing.dft_filter(self.image, filter)
        return self.image

    @reset_image
    def filter_interference(image):
        """See ImageProcessing.filter_interference"""
        self.image = ImageProcessing.filter_interference(self.image)
        return self.image



class ColorImage(Image):
    def __init__(self, image=None, path=None):
        """
            image [ndarray]: 
        """
        if not image is None:
            pass
        elif image is None and not path is None:
            image = imread(path)
        else:
            raise ImproperImageFormatException("Must include path or image array")

        if len(np.shape(image)) < 3 or np.shape(image)[2] == 1:
            raise ImproperImageFormatException()

        super().__init__(image)

    def to_gray_scale(self):
        return GrayImage(ImageProcessing.to_gray_scale(self.image))


class GrayImage(Image):
    def __init__(self, image=None, path=None):
        """
            image [ndarray]: 
        """
        if not image is None:
            pass
        elif image is None and not path is None:
            image = imread(path)
        else:
            raise ImproperImageFormatException("Must include path or image array")

        if len(np.shape(image)) > 2 and np.shape(image)[2] > 1:
            image = ImageProcessing.to_gray_scale(image)

        super().__init__(image)


class ImageProcessing(object):
    """
    A class contains static methods for use in image processing, all methods take in an image input that can either be a numpy array or Image object

    ...

    Methods
    -------
    flip(image)
        ****
    red_channel(image)
        ****
    green_channel(image)
        ****
    blue_channel(image)
        ****
    color_channel(image,color="r")
        ****
    bright_adjust(image,c)
        ****
    contrast_adjust(image,c)
        ****
    thresholder(image,c)
        ****
    cropper(image,width,height,x=0,y=0)
        ****
    scaler(image)
        ****
    light_field(image,h,w)
        ****
    plot_image(image, title="", color=True)
        ****
    to_gray_scale(image)
        ****
    to_HSB(image)
        ****
    to_RGB(image)
        ****
    alpha_blend(image1, image2, alpha=0.5)
        ****
    cross_dissolve(image1, image2, numsteps=10)
        ****
    blur_gray(image, size=3)
        ****
    median_filter_gray(image, size=3)
        ****
    convolution_gray(image, kernel)
        ****
    sharpen_gray(image)
        ****
    edge_detect_gray(image)
        ****
    compose(image, frame, transformation)
        ****
    frame(image, frame, translate=[0, 0], rotation=0, scale=1)
        ****
    dft_filter(image, filter)
        ****
    filter_interference(image)
        ****
    
    """

    @staticmethod
    @correct_image
    def flip(image):
        """Flips an image upside down

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        return image[::-1]

    @staticmethod
    @correct_image
    def red_channel(image):
        """Returns a grey value for the image with the amount of red in the image

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        return image[:, :, 0]

    @staticmethod
    @correct_image
    def green_channel(image):
        """Returns a grey value for the image with the amount of green in the image

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        return image[:, :, 1]

    @staticmethod
    @correct_image
    def blue_channel(image):
        """Returns a grey value for the image with the amount of blue in the image

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        return image[:, :, 2]

    @staticmethod
    @correct_image
    def color_channel(image, color="r"):
        """Returns a grey value for the image with the amount of the specified color in the image

        Parameters
        ----------
        image : numpy array
            the image to transform
        color : string
            the color to find the amount of from 'r', 'g', 'b' (default 'r')

        Returns
        -------
        numpy array
            the edited image
        """
        if c == "r":
            return ImageProcessing.red_channel(image)
        elif c == "g":
            return ImageProcessing.green_channel(image)
        elif c == "b":
            return ImageProcessing.blue_channel(image)
        else:
            return None

    @staticmethod
    @correct_image
    def bright_adjust(image, c, color=True):
        """Adjusts the brightness of the image by a summative factor c

        Parameters
        ----------
        image : numpy array
            the image to transform
        c : int
            amount to scale the image brightness by, chosen between -100 and 100

        Returns
        -------
        numpy array
            the edited image
        """
        if not color:
            return np.clip(image + c / 1.0, 0, 255)
        else:
            im = ImageProcessing.to_HSB(image)
            im[:, :, 2] = np.maximum(np.minimum(im[:, :, 2] + c, 255), 0)
            return ImageProcessing.to_RGB(im)

    @staticmethod
    @correct_image
    def contrast_adjust(image, c, color=True):
        """Adjusts the contrast of the image by a nonlinear factor c

        Parameters
        ----------
        image : numpy array
            the image to transform
        c : int
            amount to scale the image contrast by, chosen between -100 and 100

        Returns
        -------
        numpy array
            the edited image
        """
        if not color:
            return (1e-2 * (c + 100.0)) ** 4 * (image - 128.0) + 128.0
        else:
            im = toHSB(image)
            im[:, :, 2] = (1e-2 * (c + 100.0)) ** 4 * (im[:, :, 2] - 128.0) + 128.0
            return np.maximum(np.minimum(toRGB(im), 255), 0)

    @staticmethod
    @correct_image
    def thresholder(image, c):
        """Takes an image and makes all values above the threshold equal to 255

        Parameters
        ----------
        image : numpy array
            the image to transform
        c : int
            threshold value, must be 255 or lower

        Returns
        -------
        numpy array
            the edited image
        """
        return np.where(image < c, image, 255)

    @staticmethod
    @correct_image
    def cropper(image, width, height, x=0, y=0):
        """crops an image starting at the leftmost point x y with size width x height

        Parameters
        ----------
        image : numpy array
            the image to transform
        width : int
            the width of the cropped image
        height : int 
            the height of the cropped image
        x : int 
            the starting point of the x axis of the image (default 0)
        y : int 
            the starting point of the y axis of the image (default 0)

        Returns
        -------
        numpy array
            the edited image
        """
        return image[y : y + height, x : x + width]

    @staticmethod
    @correct_image
    def scaler(image):
        """Downsamples the pixels in the image by a factor of 2

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        return image[::2, ::2]

    @staticmethod
    @correct_image
    def light_field(image, h, w):
        """Averages a light field into its image

        Parameters
        ----------
        image : numpy array
            the light field to transform into an image
        h : int
            height of the image within the lightfield
        w : int
            width of the image within the lightfield

        Returns
        -------
        numpy array
            the edited image
        """
        rows, cols = np.shape(image)
        im_h = int(rows / h)
        im_w = int(cols / w)
        return np.sum(image.reshape(rows // im_h, im_h, -1, im_w).swapaxes(1, 2).reshape(-1, im_h, im_w), axis=0) / (
            h * w
        )

    @staticmethod
    @correct_image
    def plot_image(image, title="", color=True):
        """Plots an image using matplotlib on a figure created by the user using plt.figure()

        Parameters
        ----------
        image : numpy array
            the light field to transform into an image
        title: string
            title of the image to be displayed above the plot
        color : boolean
            true if the image is a ColorImage (default True)
        """
        im = np.array(image, dtype=np.uint8)
        if color:
            plt.imshow(im, vmin=0, vmax=255)
        else:
            plt.imshow(im, vmin=0, vmax=255, cmap="Greys_r")
        plt.title(title)

    @staticmethod
    @correct_image
    def plot_image_full(image, title="", color=True):
        """Plots an image using matplotlib on a figure created by this method

        Parameters
        ----------
        image : numpy array
            the light field to transform into an image
        title: string
            title of the image to be displayed above the plot
        color : boolean
            true if the image is a ColorImage (default True)
        """
        plt.figure()
        im = np.array(image, dtype=np.uint8)
        if color:
            plt.imshow(im, vmin=0, vmax=255)
        else:
            plt.imshow(im, vmin=0, vmax=255, cmap="Greys_r")
        plt.title(title)
        plt.show()

    @staticmethod
    @correct_image
    def to_gray_scale(image):
        """Takes in a color image (3 channels) and returns a gray scale image (2 channels)

        Parameters
        ----------
        image : numpy array
            a colored image

        Returns
        -------
        numpy array
            a gray scale image
        """
        return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

    @staticmethod
    @correct_image
    def to_HSB(image):
        """Takes in a colored RGB image and transforms it to HSB

        Parameters
        ----------
        image : numpy array
            RGB image

        Returns
        -------
        numpy array
            HSB Image
        """
        temp = 255 * colors.rgb_to_hsv(image / 255.0)
        return temp.astype(np.int32)

    @staticmethod
    @correct_image
    def to_RGB(image):
        """Takes in a colored HSB image and transforms it to RGB

        Parameters
        ----------
        image : numpy array
            HSB image

        Returns
        -------
        numpy array
            RGB image
        """
        temp = 255 * colors.hsv_to_rgb(image / 255.0)
        return temp.astype(np.int32)

    @staticmethod
    @correct_images
    def alpha_blend(image1, image2, alpha=0.5):
        """Blends to images together according to an alpha factor

        Parameters
        ----------
        image1 : numpy array
            first image to blend, has proportion alpha
        image2 : numpy array
            second image to blend, has proportion (1-alpha)
        alpha : float or numpy array
            factor to blend the values, must be between 0-1, can be a mask (default 0.5)

        Returns
        -------
        numpy array
            the blended image
        """
        return np.minimum(alpha * image1 + (1 - alpha) * image2, 255)

    @staticmethod
    @correct_images
    def cross_dissolve(image1, image2, num_steps=10):
        """Takes in 2 color images of the same size and returns an array of alpha blend of those two images

        Parameters
        ----------
        image1 : numpy array
            first image to blend, has proportion alpha
        image2 : numpy array
            second image to blend, has proportion (1-alpha)
        num_steps : int
            number of steps in the array that slowly blends the images

        Returns
        -------
        list [numpy array]
            a list of the output images starting from mostly image1 and ending with mostly image2
        """
        return [alphaBlend(image1, image2, alpha) for alpha in np.linspace(0, 1, numsteps)]

    @staticmethod
    @correct_image
    def blur_gray(image, size=3):
        """Takes in an image and returns a corresponding result that has been blurred (spatially filtered) using a uniform averaging.

        Parameters
        ----------
        image : numpy array
            the image to transform
        side : int
            length and width of the mean kernel 

        Returns
        -------
        numpy array
            the edited image
        """
        result = np.zeros(image.shape)
        edge = int((size) / 2) if size % 2 else int((size - 1) / 2)
        temp_result = np.zeros(np.array([*image.shape]) + 2 * edge)
        temp_result[edge : temp_result.shape[0] - edge, edge : temp_result.shape[1] - edge] = image[:]
        sub_images = np.zeros((image.shape[0], image.shape[1], size, size))
        for i in range(size):
            for j in range(size):
                sub_images[:, :, i, j] = temp_result[i : image.shape[0] + i, j : image.shape[1] + j]
        result = np.multiply(sub_images, np.ones((size, size))).sum(axis=2).sum(axis=2) / size ** 2
        return result

    @staticmethod
    @correct_image
    def median_filter_gray(image, size=3):
        """Takes in a grayscale image and returns a corresponding result that has been median filtered

        Parameters
        ----------
        image : numpy array
            the image to transform
        side : int
            length and width of the median kernel 

        Returns
        -------
        numpy array
            the edited image
        """
        result = np.zeros(image.shape)
        edge = int((size) / 2) if size % 2 else int((size - 1) / 2)
        temp_result = np.zeros(np.array([*image.shape]) + 2 * edge)
        temp_result[edge : temp_result.shape[0] - edge, edge : temp_result.shape[1] - edge] = image[:]
        sub_images = np.zeros((image.shape[0], image.shape[1], size, size))
        for i in range(size):
            for j in range(size):
                sub_images[:, :, i, j] = temp_result[i : image.shape[0] + i, j : image.shape[1] + j]
        result = np.median(
            np.multiply(sub_images, np.ones((size, size))).reshape((image.shape[0], image.shape[1], size ** 2)), axis=2
        )

        return result

    @staticmethod
    @correct_image
    def convolution_gray(image, kernel):
        """General convolution function that takes in an image and kernel, and performs the appropriate convolution

        Parameters
        ----------
        image : numpy array
            the image to transform
        kernel : numpy array
            the kernel to convolve with

        Returns
        -------
        numpy array
            the edited image
        """
        result = np.zeros(image.shape)
        kernel_temp = np.squeeze(np.asarray(kernel.T))
        temp_result = np.zeros(
            np.array([*image.shape])
            + np.array([2 * int((kernel_temp.shape[0]) / 2), 2 * int((kernel_temp.shape[1]) / 2)])
        )
        temp_result[
            int((kernel_temp.shape[0]) / 2) : temp_result.shape[0] - int((kernel_temp.shape[0]) / 2),
            int((kernel_temp.shape[1]) / 2) : temp_result.shape[1] - int((kernel_temp.shape[1]) / 2),
        ] = image[:]
        sub_images = np.zeros((image.shape[0], image.shape[1], kernel_temp.shape[0], kernel_temp.shape[1]))
        for i in range(kernel_temp.shape[0]):
            for j in range(kernel_temp.shape[1]):
                sub_images[:, :, i, j] = temp_result[i : image.shape[0] + i, j : image.shape[1] + j]
        result = np.multiply(sub_images, kernel_temp).sum(axis=2).sum(axis=2)
        return result

    @staticmethod
    @correct_image
    def sharpen_gray(image):
        """Takes in a grayscale image and returns a corresponding result that has been sharpened

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        kernel = np.matrix([[0, -1, 0], [-1, 6, -1], [0, -1, 0]]) / 2.0
        return np.maximum(np.minimum(ImageProcessing.convolution_gray(image, kernel), 255), 0)

    @staticmethod
    @correct_image
    def edge_detect_gray(image):
        """Takes in a grayscale image and returns a corresponding result that shows the gradient magnitude of the input

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        sobel_x_filter = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y_filter = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
        sobel_x = ImageProcessing.convolution_gray(image, sobel_x_filter) / 8.0
        sobel_y = ImageProcessing.convolution_gray(image, sobel_y_filter) / 8.0
        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return gradient

    @staticmethod
    @correct_image
    def compose(image, frame, transformation):
        """Takes in an image and a frame and the transformation to compose the two together

        Parameters
        ----------
        image : numpy array
            the image to transform
        frame : numpy array
            the image frame
        transformation : numpy array
            the transformation matrix

        Returns
        -------
        numpy array
            the edited image
        """

        width, height = frame.size

        # Invert matrix for compose function, grab values for Affine Transform
        t = np.linalg.inv(transformation)
        a = t[0, 0]
        b = t[0, 1]
        c = t[0, 2]
        d = t[1, 0]
        e = t[1, 1]
        f = t[1, 2]

        image = image.transform((width, height), Image.AFFINE, (a, b, c, d, e, f), Image.BICUBIC)

        # Make mask from image's location
        im = np.sum(np.asarray(image), -1)
        vals = 255.0 * (im > 0)
        mask = Image.fromarray(vals).convert("1")

        # Composite images together
        result = Image.composite(image, frame, mask)

        return result

    @staticmethod
    @correct_image
    def frame(image, frame, translate=[0, 0], rotation=0, scale=1):
        """Takes in an image and a frame and composes them together

        Parameters
        ----------
        image : numpy array
            the image to transform
        frame : numpy array
            the image frame
        translate : list(int)
            the x, y coordinates in which to translate on the frame (default:[0,0])
        rotation : int/float
            angle to rotate image in degrees (default:0)
        scale : int/float 
            the factor to scale the image by (default:1)

        Returns
        -------
        numpy array
            the edited image
        """
        translate = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])
        rotation = np.array(
            [
                [np.cos(rotation / 180 * np.pi), np.sin(rotation / 180 * np.pi), 0],
                [-np.sin(rotation / 180 * np.pi), np.cos(rotation / 180 * np.pi), 0],
                [0, 0, 1],
            ]
        )
        scale = np.diag([scale, scale, 1])

        return ImageProcessing.compose(image, frame, np.matmul(np.matmul(translate, rotation), scale))

    @staticmethod
    @correct_image
    def dft_filter(image, filter):
        """Takes in an image and a filter and computes the fft and filters via multiplying with the filter

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """
        F = np.fft.fft2(np.array(image, dtype="complex"))
        X, Y = np.meshgrid(np.linspace(-10, 10, image.shape[0]), np.linspace(-10, 10, image.shape[1]))
        mu, sigma = 0, 1
        H = filter
        H = np.fft.fftshift(H)
        G = np.zeros(F.shape, dtype="complex")
        G = F * H
        g = np.fft.ifft2(G)
        return np.minimum(np.maximum(np.real(g), 0), 255)

    @staticmethod
    @correct_image
    def filter_interference(image):
        """Takes in an image and filters out any stray frequency

        Parameters
        ----------
        image : numpy array
            the image to transform

        Returns
        -------
        numpy array
            the edited image
        """        
        F = np.fft.fft2(np.array(image, dtype='complex'))

        neighboring=np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,-1]])
        G = ImageProcessing.convolution(np.abs(F),neighboring) 

        blur_kernel = np.matrix([[1, 1, 1],[1, 1, 1],[1, 1, 1]]) / 9.0
        average = ImageProcessing.convolution(F, blur_kernel)

        G = np.where(G > 0, average, F)

        g = np.fft.ifft2( G )
        return np.minimum(np.maximum(np.real(g),0),255)




