// Entry point for the unpkg bundle containing custom model definitions.
//
// It differs from the notebook bundle in that it does not need to define a
// dynamic baseURL for the static assets and may load some css that would
// already be loaded by the notebook otherwise.

// Export widget models and views, and the npm package version number.
var _ = require("underscore");
module.exports = _.extend({},
    require("./exatomic-base.js"),
    require("./exatomic-utils.js"),
    require("./exatomic-three.js"),
    require("./exatomic-widgets.js"),
    require("./exatomic-editor.js")
);
module.exports["version"] = require("../package.json").version;
