from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[
    Extension("subroutine_cython",
              ["subroutine_cython.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math","-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp'],
              include_dirs = [np.get_include()]
              ) 
]

setup( 
  name = "subroutine_cython",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)