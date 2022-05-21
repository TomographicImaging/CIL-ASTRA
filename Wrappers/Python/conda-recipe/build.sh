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

set -x
mkdir ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../cil/" ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../test/" ${SRC_DIR}/ccpi
cp -v ${RECIPE_DIR}/../setup.py ${SRC_DIR}/ccpi

cd ${SRC_DIR}/ccpi
echo "Python command is ${PYTHON}"

pip install .
