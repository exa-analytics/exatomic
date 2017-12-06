// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/**
 * Tests for Base Widgets
 */
"use strict";
var QUnit = require("qunitjs");
var _ = require("underscore");
var widget = require("../widget.js");


/** Model class for testing bidirectional communication. */
class TestDOMWidgetModel extends widget.DOMWidgetModel {
    /**
     * Get the default class values.
     * Used by jupyter-js-widgets.
     */
    get defaults() {
        return _.extend({}, widget.DOMWidgetModel.prototype.defaults, {
            _view_name: "TestDOMWidgetView",
            _model_name: "TestDOMWidgetModel",
            frontend_text: "Hello World!",
            backend_counter: 0
        });
    }
}


/** View class for testing bidirectional communication. */
class TestDOMWidgetView extends widget.DOMWidgetView {
    /**
     * Render the view in the notebook.
     * Call order it initialize, constructor, render.
     */
    render() {
        this.set_frontend();
        this.listenTo(this.model, "change:frontend_text", this.change, this);
    }

    /**
     * Set the value of the widget text
     */
    set_frontend() {
        this.el.textContent = this.model.get("frontend_text");
    }

    /**
     * Call front end setter and then backend incrementer.
     */
    change() {
        this.set_frontend();
        this.inc_backend();
    }

    /**
     * Increment backend counter and set the model.
     */
    inc_backend() {
        var bc = this.model.get("backend_counter");
        bc++;
        this.model.set("backend_counter", bc);
        this.model.save_changes();
        //this.touch();
    }
}


/** Mock model for exa.core.tests.test_widget.Kontainer */
class TestKontainerModel extends TestDOMWidgetModel {
    /**
     * Get the default class values.
     */
    get defaults() {
        return _.extend({}, TestDOMWidgetModel.prototype.defaults, {
            _view_name: "TestKontainerView",
            _model_name: "TestKontainerModel",
        });
    }
}

/** Mock view for exa.core.tests.test_widget.Kontainer */
class TestKontainerView extends TestDOMWidgetView {}



module.exports = {
    TestDOMWidgetModel: TestDOMWidgetModel,
    TestDOMWidgetView: TestDOMWidgetView,
    TestKontainerModel: TestKontainerModel,
    TestKontainerView: TestKontainerView
};
