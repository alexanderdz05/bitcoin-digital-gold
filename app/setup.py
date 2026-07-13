"""
Build the monte_carlo Pybind11 extension.

From the app/ directory, run:
    pip install pybind11
    python setup.py build_ext --inplace

This produces monte_carlo.cpython-3XX-darwin.so (Mac) or
monte_carlo.cpython-3XX-win_amd64.pyd (Windows) in the same directory.
import it as: import monte_carlo
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
    name="monte_carlo",
    sources=["monte_carlo.cpp"],
    # Uncomment for Linux OpenMP:
    # extra_compile_args=["-O3", "-march=native", "-std=c++17", "-fopenmp"],
    # extra_link_args=["-fopenmp"],
    # Uncomment for Mac OpenMP (after: brew install libomp):
    extra_compile_args=["-O3", "-march=native", "-std=c++17", "-Xpreprocessor", "-fopenmp"],
    extra_link_args=["-lomp"],
)

setup(
    name="monte_carlo",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)