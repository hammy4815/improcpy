===============================
Installation
===============================

The following are the steps to install improcpy:

#. Step 1. improcpy can be imported very easily. First clone the `GitHub`_ repo

    .. _GitHub: https://github.com/hammy4815/improcpy

    .. code-block:: bash

        git clone git@github.com:hammy4815/improcpy.git

#. Step 2. Ensure `pip`_ is installed on your machine

    .. _pip: https://pip.pypa.io/en/stable/installing

#. Step 3. Install locally via pip

    .. code-block:: bash

        pip install -e .

#. Step 4. You can now import the different components of improcpy into your python module

    .. code-block:: python

        from improcpy import ImageProcessing as ip 
        from improcpy import GrayImage, ColorImage