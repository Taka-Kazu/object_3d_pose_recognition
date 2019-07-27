#!/usr/bin/env python
#! coding utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Options

import numpy

Options.annotate = True

ext_modules = [Extension("euclidean_clustering", ["euclidean_clustering.pyx"], include_dirs=[numpy.get_include()], language="c++"),
]

setup(cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
