$PYTHON setup.py install
jupyter nbextension install --py --symlink --sys-prefix exatomic
jupyter nbextension enable --py --sys-prefix exatomic
