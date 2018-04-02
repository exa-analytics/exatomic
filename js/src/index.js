// Entry point for the notebook bundle containing custom model definitions.
//
// Setup notebook base URL
//
// Some static assets may be required by the custom widget javascript. The base
// url for the notebook is not known at build time and is therefore computed
// dynamically.
// var __webpack_public_path__ = document.querySelector("body").getAttribute("data-base-url") + "nbextensions/exatomic/";

// Export widget models and views, and the npm package version number.
module.exports = {};

var loaded = [
    require("./base"),
    require("./utils"),
    require("./appthree"),
    require("./widgets"),
    require("./tensor")
];

for (var m in loaded) {
    if (loaded.hasOwnProperty(m)) {
        var mod = loaded[m];
        for (var item in mod) {
            if (mod.hasOwnProperty(item)) {
                module.exports[item] = mod[item];
            };
        };
    };
};

// var _ = require("underscore");
//module.exports = _.extend({},
//    require("./base.js"),
//    require("./utils.js"),
//    require("./appthree.js"),
//    require("./widgets.js"),
//    require("./tensor.js")
//);
module.exports["version"] = require("../package.json").version;
