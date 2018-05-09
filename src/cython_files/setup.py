from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


libs = []
args = ['-O3']
sources = ['cython_integrand.pyx']
include = [numpy.get_include()]
linkerargs = ['-Wl,-rpath,$(PWD)/lib']


extensions = [
    Extension("cython_integrand",
              sources=sources,
              include_dirs=include,
              libraries=libs,
              extra_compile_args=args)
]

setup(name='cython_integrand',
      packages=['cython_integrand'],
      ext_modules=cythonize(extensions),
      )
