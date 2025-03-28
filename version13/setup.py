from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform


if platform.system() == 'Windows':
    extra_compile_args = ['/O2', '/fp:fast', '/arch:AVX2']
else:
    extra_compile_args = ['-O3', '-march=native', '-ffast-math']
    extra_compile_args.append('-fopenmp')
    extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "move",
        ["move.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()]
    ),
    Extension(
        "next_move_generator",
        ["next_move_generator.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()]
    ),
    Extension(
        "AI",
        ["AI.pyx"],
        extra_compile_args=extra_compile_args,
        include_dirs=[np.get_include()]
    )
]

setup(
    name="abalone_cython",
    version="1.0.0",
    author="Park",
    description="Abalone game AI using Cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': True,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
            'nonecheck': False,
        }
    ),
    include_dirs=[np.get_include()]
)