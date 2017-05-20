REM Helper to build reStructuredText (RST) files using jsdoc
REM Run from the js\ directory
REM Run 'npm install' first if necessary
IF NOT EXIST node_modules\jsdoc-sphinx mkdir node_modules & git clone https://github.com/HumanBrainProject/jsdoc-sphinx node_modules/jsdoc-sphinx
CALL node_modules\.bin\jsdoc -t node_modules\jsdoc-sphinx\template\ -d ..\docs\source\js\jsdoc_rst\  src\exa-abcwidgets.js
CALL del ..\docs\source\js\jsdoc_rst\index.rst 
CALL del ..\docs\source\js\jsdoc_rst\conf.py
