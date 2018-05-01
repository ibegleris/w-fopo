
from distutils.core import setup
from Cython.Build import cythonize
import numpy
"""
setup(
    ext_modules=cythonize('cython_integrand.pyx'),
    include_dirs=[numpy.get_include()]
)   
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
setup(
  name = 'cython_integrand',
  ext_modules=[
    Extension('cython_integrand',
              sources=['cython_integrand.pyx'],
              extra_compile_args=['-O3', '-fmove-loop-invariants'],
              language='c')
    ],
  cmdclass = {'build_ext': build_ext},
  include_dirs=[numpy.get_include()]
)
