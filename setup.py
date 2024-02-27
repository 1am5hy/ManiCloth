import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='diffcloth_py',
    version='0.0.1',
    long_description='',
    ext_modules=[CMakeExtension('diffcloth_py')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
