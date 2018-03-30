"%PYTHON%" setup.py install --single-version-externally-managed --record=record.txt
jupyter nbextension enable exatomic --py --sys-prefix
if errorlevel 1 exit 1
