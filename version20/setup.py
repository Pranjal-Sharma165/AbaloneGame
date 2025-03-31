from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "move_cy",
        ["move_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3']
    ),
    Extension(
        "next_move_generator_cy",
        ["next_move_generator_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3']
    ),
    Extension(
        "AI",
        ["AI.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    ext_modules=cythonize(extensions)
)