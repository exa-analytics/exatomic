<<<<<<< HEAD
// Entry point for the unpkg bundle containing custom model definitions.
//
// It differs from the notebook bundle in that it does not need to define a
// dynamic baseURL for the static assets and may load some css that would
// already be loaded by the notebook otherwise.

// Export widget models and views, and the npm package version number.
var exports = require("./exports.js");
module.exports = exports.required;
=======
// Entry point for the unpkg bundle containing custom model definitions.
//
// It differs from the notebook bundle in that it does not need to define a
// dynamic baseURL for the static assets and may load some css that would
// already be loaded by the notebook otherwise.

// Export widget models and views, and the npm package version number.
var _ = require("underscore");
module.exports = _.extend({},
    require("./jupyter-exatomic-base.js"),
    require("./jupyter-exatomic-utils.js"),
    require("./jupyter-exatomic-three.js"),
    require("./jupyter-exatomic-widgets.js"),
    require("./jupyter-exatomic-examples.js")
);
module.exports["version"] = require("../package.json").version;
>>>>>>> 454ebaa9677a776a535abb28f528efefabda52c5
