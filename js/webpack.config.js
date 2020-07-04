var path = require("path");
var version = require("./package.json").version;



module.exports = [
    {// Notebook extension
        entry: "./src/extension.js",
        output: {
            filename: "extension.js",
            path: path.resolve(__dirname, "..", "exatomic", "static", "js"),
            libraryTarget: "amd"
        }
    },
    {// exatomic bundle for the classic notebook
        entry: "./src/notebook.js",
        output: {
            filename: "index.js",
            path: path.resolve(__dirname, "..", "exatomic", "static", "js"),
            libraryTarget: "amd"
        },
        devtool: "source-map",
        externals: ["@jupyter-widgets/base", "@jupyter-widgets/controls"]
    },
    {// exatomic bundle for unpkg
        entry: "./src/embed.js",
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
