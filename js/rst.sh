#!/usr/bin/env bash
# Helper to build reStructuredText (RST) files using jsdoc
# Run from the js/ directory
# Run 'npm install' first if necessary
if [[ ! -d node_modules/jsdoc-sphinx ]]
then
    mkdir -p node_modules
    git clone https://github.com/HumanBrainProject/jsdoc-sphinx node_modules/jsdoc-sphinx
fi
./node_modules/.bin/jsdoc -t node_modules/jsdoc-sphinx/template/ -d ../docs/source/js/jsdoc_rst/ -r src/
rm ../docs/source/js/jsdoc_rst/index.rst ../docs/source/js/jsdoc_rst/conf.py
