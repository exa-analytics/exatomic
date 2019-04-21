#!/usr/bin/env bash
#rm -rf node_modules/jsdoc_sphinx
npm run build
npm install
jupyter nbextension install exatomic --py --sys-prefix --overwrite
jupyter nbextension enable exatomic --py --sys-prefix
