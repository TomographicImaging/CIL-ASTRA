set -x
mkdir ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../cil/" ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../test/" ${SRC_DIR}/ccpi
cp -v ${RECIPE_DIR}/../setup.py ${SRC_DIR}/ccpi

cd ${SRC_DIR}/ccpi
echo "Python command is ${PYTHON}"

pip install .
