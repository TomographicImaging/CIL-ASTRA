#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import setup
import os
import subprocess

cil_version = subprocess.check_output('git describe', shell=True).decode("utf-8").rstrip()

print("CIL VERSION = {}".format(cil_version))

if os.environ.get('CONDA_BUILD', 0) == 0:
      cwd = os.getcwd()
else:
      cwd = os.path.join(os.environ.get('RECIPE_DIR'),'..')

fname = os.path.join(cwd, 'cil', 'plugins', 'astra', 'version.py')

if os.path.exists(fname):
    os.remove(fname)
with open(fname, 'w') as f:
    f.write('version = \'{}\''.format(cil_version))


setup(
    name="cil-astra",
    version=cil_version,
    packages=['cil.plugins.astra',
              'cil.plugins.astra.operators',
              'cil.plugins.astra.processors',
              'cil.plugins.astra.utilities'],

    # metadata for upload to PyPI
    author="Edoardo Pasca",
    author_email="edoardo.pasca@stfc.ac.uk",
    description='CCPi Core Imaging Library - Astra bindings',
    license="OSI Approved :: Apache Software License",
    keywords="Python Framework",
    url="http://www.ccpi.ac.uk/cil",   # project home page, if any
)
