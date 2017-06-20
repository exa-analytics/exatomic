REM Helper to build reStructuredText (RST) files using jsdoc
REM Run from the js\ directory
REM Run 'npm install' first if necessary
CALL del /f /s /q ..\docs\source\js 1>nul
CALL rmdir /s /q ..\docs\source\js
CALL npm install jsdoc
CALL npm install jsdoc-sphinx
CALL node_modules\.bin\jsdoc -t node_modules\jsdoc-sphinx\template\ -d ..\docs\source\js\ -r src\
CALL del /s /q ..\docs\source\js\jsdoc_rst\conf.py
