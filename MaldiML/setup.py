#!/usr/bin/env python

from distutils.core import setup

setup(name='MaldiML',
      version='0.1',
      description='MaldiML package',
      author='Caroline Weis',
      author_email='carolineweis@gmx.de',
      license='Apache License 2.0',
      packages=['MaldiML'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
      ]
)

