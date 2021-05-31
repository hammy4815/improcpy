from setuptools import setup

setup(
    name='improcpy',
    version='0.1.0',    
    description='Image Processing Python',
    url='https://github.com/hammy4815/improcpy',
    author='Ian Hammond',
    author_email='hammy4815@gmail.com',
    license='MIT',
    packages=['improcpy'],
    install_requires=['matplotlib',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)