from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="abalone_cython",
    ext_modules=cythonize([
        "AI.py",
        "move.py",
        "next_move_generator.py"
    ]),
    include_dirs=[np.get_include()]
)