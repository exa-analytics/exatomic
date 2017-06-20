CALL del /q /s /q node_modules/jsdoc_sphinx 1>nul
CALL rmdir /s /q node_modules/jsdoc_sphinx
CALL npm install
CALL jupyter nbextension install exatomic --py --sys-prefix --overwrite
CALL jupyter nbextension enable exatomic --py --sys-prefix
