===============================
ColorImage
===============================

This object inherits from Image and represents a color image.

Class
===============================

.. autoclass:: improcpy.image.ColorImage
   :members:


Examples
===============================

.. code-block:: python

   from improcpy.image import ColorImage

   image = ColorImage(path="sample_images/worthen.jpg")
   image.plot_image_full(title="worthen", path="sample_tested/worthen.png")

.. image:: ../improcpy/sample_tested/worthen.png
   :width: 400
   :alt: Alternative text

.. code-block:: python

   image.flip()
   image.plot_image_full(title="worthen flipped", path="sample_tested/worthen_flipped.png")

.. image:: ../improcpy/sample_tested/worthen_flipped.png
   :width: 400
   :alt: Alternative text

.. code-block:: python

   new_image = image.cropper(250, 250, x=50, y=50, reset=True)
   new_image.plot_image_full(title="worthen partial", path="sample_tested/worthen_partial.png")

.. image:: ../improcpy/sample_tested/worthen_partial.png
   :width: 400
   :alt: Alternative text

.. code-block:: python

   image.plot_image_full(title="worthen unchanged", path="sample_tested/worthen_unchanged.png")

.. image:: ../improcpy/sample_tested/worthen_unchanged.png
   :width: 400
   :alt: Alternative text

.. code-block:: python

   image.hard_reset()
   image.plot_image_full(title="worthen restored", path="sample_tested/worthen_restored.png")

.. image:: ../improcpy/sample_tested/worthen_restored.png
    :width: 400
    :alt: Alternative text