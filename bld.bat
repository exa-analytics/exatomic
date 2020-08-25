"%PYTHON%" -m pip install . -vv
jupyter nbextension enable exatomic --py --sys-prefix
if errorlevel 1 exit 1
