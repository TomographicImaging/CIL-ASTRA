if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi  
set -ex
mkdir ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../cil/" ${SRC_DIR}/ccpi
cp -rv "${RECIPE_DIR}/../test/" ${SRC_DIR}/ccpi
cp -v ${RECIPE_DIR}/../setup.py ${SRC_DIR}/ccpi

cd ${SRC_DIR}/ccpi
echo "Python command is ${PYTHON}"
${PYTHON} setup.py install
