// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=================
jupyter-exatomic.js
=================
JavaScript "frontend" complement of exatomic's Container for use within
the Jupyter notebook interface. This "module" standardizes bidirectional
communication logic for all container widget views.
*/

"use strict";
var _ = require("underscore");
var widgets = require("@jupyter-widgets/base");
var control = require("@jupyter-widgets/controls");
var utils = require("./jupyter-exatomic-utils.js");
var App3D = require("./jupyter-exatomic-three.js").App3D;


var ExatomicBoxModel = control.BoxModel.extend({

    defaults: function() {
        return _.extend({}, control.BoxModel.prototype.defaults, {
            _model_module: "jupyter-exatomic",
            _view_module: "jupyter-exatomic",
            _model_name: "ExatomicBoxModel",
            _view_name: "ExatomicBoxView",
        })
    }

});

var ExatomicBoxView = control.BoxView.extend({

});


var ExatomicSceneModel = widgets.DOMWidgetModel.extend({

    defaults: function() {
        return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_module: "jupyter-exatomic",
            _view_module: "jupyter-exatomic",
            _model_name: "ExatomicSceneModel",
            _view_name: "ExatomicSceneView",
            scn_clear: false,
            scn_saves: false,
            field_iso: 2.0,
            field_ox: -3.0,
            field_oy: -3.0,
            field_oz: -3.0,
            field_dx: 0.2,
            field_dy: 0.2,
            field_dz: 0.2,
            field_nx: 31,
            field_ny: 31,
            field_nz: 31,
            savedir: "",
            imgname: "",
        })
    }

});


var ExatomicSceneView = widgets.DOMWidgetView.extend({

    initialize: function() {
        ExatomicSceneView.__super__.initialize.apply(this, arguments);
        var that = this;
        $(this.el).width(
            this.model.get("layout").get("width")).height(
            this.model.get("layout").get("height")).resizable({
            aspectRatio: false,
            resize: function(event, ui) {
                var w = ui.size.width;
                var h = ui.size.height;
                that.model.get("layout").set("width", w);
                that.model.get("layout").set("height", h);
                that.el.width = w;
                that.el.height = h;
                that.resize(w, h);
            }
        });
        this.init();
        this.init_listeners();
        this.field_listeners();
    },

    init: function() {
        this.meshes = {"generic": [],
                       "frame": [],
                       "field": [],
                       "atom": [],
                       "two": []};
        this.app3d = new App3D(this);
    },

    render: function() {
        this.renderer.render(this.scene, this.camera);
    },

    resize: function(w, h) {
        this.model.get("layout").set("width", w);
        this.model.get("layout").set("height", h);
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.controls.handleResize();
        this.render();
    },

    animation: function() {
        window.requestAnimationFrame(this.animation.bind(this));
        this.controls.update();
        this.resize(this.model.get("layout").get("width"),
                    this.model.get("layout").get("height"));
    },

    clear_meshes: function(kind) {
        kind = (typeof kind !== "string") ? "all" : kind;
        for (var idx in this.meshes) {
            if ((kind === "all") || (kind === idx)) {
                for (var sub in this.meshes[idx]) {
                    this.scene.remove(this.meshes[idx][sub]);
                    delete this.meshes[idx][sub];
                };
            };
        };
    },

    add_meshes: function(kind) {
        kind = (typeof kind !== "string") ? "all" : kind;
        for (var idx in this.meshes) {
            if ((kind === "all") || (kind === idx)) {
                for (var sub in this.meshes[idx]) {
                    this.scene.add(this.meshes[idx][sub]);
                };
            };
        };
    },

    scene_save: function() {
        this.renderer.setSize(1920, 1080);
        this.camera.aspect = 1920 / 1080;
        this.camera.updateProjectionMatrix();
        this.render();
        var image = this.renderer.domElement.toDataURL("image/png");
        this.send({"type": "image", "content": image});
        this.renderer.setSize(this.model.get("layout").get("width"),
                              this.model.get("layout").get("height"));
        var ar = this.model.get("layout").get("width") / this.model.get("layout").get("height");
        this.camera.aspect = ar;
        this.camera.updateProjectionMatrix();
        this.render();
    },

    get_fps: function() {
        return {ox: this.model.get("field_ox"),
                oy: this.model.get("field_oy"),
                oz: this.model.get("field_oz"),
                nx: this.model.get("field_nx"),
                ny: this.model.get("field_ny"),
                nz: this.model.get("field_nz"),
                fx: this.model.get("field_fx"),
                fy: this.model.get("field_fy"),
                fz: this.model.get("field_fz")}
    },

    add_field: function() {},

    field_listeners: function() {
        this.listenTo(this.model, "change:field_nx", this.add_field);
        this.listenTo(this.model, "change:field_ny", this.add_field);
        this.listenTo(this.model, "change:field_nz", this.add_field);
        this.listenTo(this.model, "change:field_iso", this.add_field);
    },

    init_listeners: function() {
        this.listenTo(this.model, "change:scn_clear", this.clear_meshes);
        this.listenTo(this.model, "change:scn_saves", this.scene_save);
    },

});


module.exports = {
    ExatomicSceneModel: ExatomicSceneModel,
    ExatomicSceneView: ExatomicSceneView,
    ExatomicBoxModel: ExatomicBoxModel,
    ExatomicBoxView: ExatomicBoxView
}

