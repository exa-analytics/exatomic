"%PYTHON%" setup.py install
jupyter nbextension enable exatomic --py --sys-prefix
if errorlevel 1 exit 1
