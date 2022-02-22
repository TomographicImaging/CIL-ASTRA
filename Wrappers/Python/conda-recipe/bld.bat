ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%"

pip install .
if errorlevel 1 exit 1
