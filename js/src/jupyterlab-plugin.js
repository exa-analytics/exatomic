// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0

var exatomic = require("./index");

var base = require("@jupyter-widgets/base");

module.exports = {
    id: "exatomic",
    requires: [base.IJupyterWidgetRegistry],
    activate: function(app, widgets) {
        widgets.registerWidget({
            name: "exatomic",
            version: exatomic.version,
            exports: exatomic
        });
    },
    autoStart: true
};
