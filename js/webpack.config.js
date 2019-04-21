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
                chunks: 'async',
                minChunks: 1,
                minSize: 30000,
                name: true
            }
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
        externals: ["@jupyter-widgets/base", "@jupyter-widgets/controls"]
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
        externals: ["@jupyter-widgets/base", "@jupyter-widgets/controls"]
    }
];
