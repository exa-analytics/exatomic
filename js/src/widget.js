// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/**
 * This module provides the base model and view used by
 * exa.core.widget.DOMWidget.
 * @module
 */
"use strict";
var ipywidgets = require("jupyter-js-widgets");
var _ = require("underscore");


/**
 * Model class for all "Exa" DOMWidgets.
 */
class DOMWidgetModel extends ipywidgets.DOMWidgetModel {
    /**
     * Get the default class values.
     * Used by jupyter-js-widgets.
     */
    get defaults() {
        return _.extend({}, ipywidgets.DOMWidgetModel.prototype.defaults, {
            _view_name: "DOMWidgetView",
            _view_module: "jupyter-exa",
            _view_module_version: "^0.4.0",
            _model_name: "DOMWidgetModel",
            _model_module: "jupyter-exa",
            _model_module_version: "^0.4.0"
        });
    }
}


/**
 * View class for all "Exa" DOMWidgets.
 */
class DOMWidgetView extends ipywidgets.DOMWidgetView {
}


module.exports = {
    DOMWidgetModel: DOMWidgetModel,
    DOMWidgetView: DOMWidgetView
};
