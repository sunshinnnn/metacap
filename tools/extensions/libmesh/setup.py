from setuptools import setup
from Cython.Build import cythonize
# try:
#     from setuptools import setup
# except ImportError:
#     from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

setup(name = 'libmesh',
      ext_modules = cythonize("*.pyx"),
      include_dirs=[numpy.get_include()])
