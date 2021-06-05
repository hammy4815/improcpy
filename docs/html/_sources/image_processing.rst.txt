===============================
ImageProcessing
===============================

This class contains static methods that act as stand alone functions. They perform image processing routines on images and are used by the Image object

Class
===============================

.. autoclass:: improcpy.image.ImageProcessing
    :members:

Examples
===============================

.. code-block:: python

    from matplotlib.pyplot import imread
    from improcpy.image import ImageProcessing as ip
    image = imread("sample_images/cosmo.jpg")
    ip.plot_image_full(image, "cosmo", path="sample_tested/cosmo")

.. image:: ../improcpy/sample_tested/cosmo.png
    :width: 400
    :alt: Alternative text

.. code-block:: python

    image = ip.to_gray_scale(image)
    ip.plot_image_full(image, "gray cosmo", path="sample_tested/cosmo_gray", color=False)

.. image:: ../improcpy/sample_tested/cosmo_gray.png
    :width: 400
    :alt: Alternative text

.. code-block:: python 

    image = GrayImage(path="sample_images/cosmo.jpg")
    X, Y = np.meshgrid(np.linspace(-10, 10, image.shape[1]), np.linspace(-10, 10, image.shape[0]))
    H = np.exp(-(X ** 2 + Y ** 2) / (2.0 * 0.5 ** 2))
    image = ip.dft_filter(image, H)
    ip.plot_image_full(image, "blurry cosmo", path="sample_tested/cosmo_blend", color=False)

.. image:: ../improcpy/sample_tested/cosmo_blend.png
    :width: 400
    :alt: Alternative text