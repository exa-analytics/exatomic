// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/**
 * Exported Modules
 */
"use strict";
var _ = require("underscore");


var modules = _.extend(
    {},
    require("./widget.js")
);

var tests = _.extend(
    {},
    require("./tests/test_widget.js")
);


module.exports = {
    modules: modules,
    tests: tests,
    required: _.extend(
        {},
        modules,
        tests,
        {"version": require("../package.json").version})
};
