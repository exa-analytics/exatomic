const webpack = require("webpack");
const path = require("path");
var version = require("./package.json").version;

/*
 * SplitChunksPlugin is enabled by default and replaced
 * deprecated CommonsChunkPlugin. It automatically identifies modules which
 * should be splitted of chunk by heuristics using module duplication count and
 * module category (i. e. node_modules). And splits the chunks…
 *
 * It is safe to remove "splitChunks" from the generated configuration
 * and was added as an educational example.
 *
 * https://webpack.js.org/plugins/split-chunks-plugin/
 *
 */

/*
 * We've enabled UglifyJSPlugin for you! This minifies your app
 * in order to load faster and run less javascript.
 *
 * https://github.com/webpack-contrib/uglifyjs-webpack-plugin
 *
 */

const UglifyJSPlugin = require('uglifyjs-webpack-plugin');



module.exports = [
    {
        entry: "./src/extension.js",
        mode: "development",
	    output: {
		    filename: "extension.js",
            path: path.resolve(__dirname, "..", "exatomic", "static", "js"),
            libraryTarget: "amd"
        },
        devtool: "source-map",
    	optimization: {
    		splitChunks: {
    			cacheGroups: {
    				vendors: {
    					priority: -10,
    					test: /[\\/]node_modules[\\/]/
    				}
    			},
    
    			chunks: 'async',
    			minChunks: 1,
    			minSize: 30000,
    			name: true
    		}
    	},
        module: {
    		rules: [
    			{
    				include: [path.resolve(__dirname, 'src')],
    				loader: 'babel-loader',
    
    				options: {
    					plugins: ['syntax-dynamic-import'],
    
    					presets: [
    						[
    							'@babel/preset-env',
    							{
    								modules: "amd"
    							}
    						]
    					]
    				},
    
    				test: /\.js$/
    			}
		    ],
         }
	},
    {
        entry: "./src/index.js",
        mode: "production",
	    output: {
		    filename: "index.js",
            path: path.resolve(__dirname, "..", "exatomic", "static", "js"),
            libraryTarget: "amd"
        },
        devtool: "source-map",
        externals: ["@jupyter-widgets/base", "@jupyter-widgets/controls"],
        module: {
    		rules: [
    			{
    				include: [path.resolve(__dirname, 'src')],
    				loader: 'babel-loader',
    
    				options: {
    					plugins: ['syntax-dynamic-import'],
    
    					presets: [
    						[
    							'@babel/preset-env',
    							{
    								modules: "amd"
    							}
    						]
    					]
    				},
    
    				test: /\.js$/
    			}
		    ],
         }
    },
    {
        entry: "./src/embed.js",
        mode: "production",
        output: {
            filename: "index.js",
            path: path.resolve(__dirname, "dist"),
            libraryTarget: "amd",
            publicPath: "https://unpkg.com/exatomic@" + version + "/dist/"
        },
        devtool: "source-map",
        externals: ["@jupyter-widgets/base", "@jupyter-widgets/controls"],
        module: {
    		rules: [
    			{
    				include: [path.resolve(__dirname, 'src')],
    				loader: 'babel-loader',
    
    				options: {
    					plugins: ['syntax-dynamic-import'],
    
    					presets: [
    						[
    							'@babel/preset-env',
    							{
    								modules: "amd"
    							}
    						]
    					]
    				},
    
    				test: /\.js$/
    			}
		    ],
         }
    }
];
