#rm -rf node_modules/jsdoc-sphinx
#git clone http://github.com/HumanBrainProject/jsdoc-sphinx node_modules/jsdoc-sphinx
rm -rf ../docs/source/js
npm install jsdoc
npm install jsdoc-sphinx
mkdir -p ../docs/source/js
./node_modules/.bin/jsdoc -t node_modules/jsdoc-sphinx/template/ -d ../docs/source/js/ -r src/
rm ../docs/source/js/conf.py
