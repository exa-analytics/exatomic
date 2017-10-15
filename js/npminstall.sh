#rm -rf node_modules/jsdoc_sphinx
npm install
jupyter nbextension install exatomic --py --sys-prefix --overwrite
jupyter nbextension enable exatomic --py --sys-prefix
