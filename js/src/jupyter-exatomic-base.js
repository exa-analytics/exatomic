// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=================
jupyter-exatomic-base.js
=================
JavaScript "frontend" complement of exatomic's Universe
for use within the Jupyter notebook interface.
*/

"use strict";
var widgets = require("@jupyter-widgets/base");
var control = require("@jupyter-widgets/controls");
var App3D = require("./jupyter-exatomic-three.js").App3D;
var version = "~" + require("../package.json").version;


var ExatomicBoxModel = control.BoxModel.extend({

    defaults: function() {
        return _.extend({}, control.BoxModel.prototype.defaults, {
            _model_module_version: version,
            _view_module_version: version,
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
            _model_module_version: version,
            _view_module_version: version,
            _model_module: "jupyter-exatomic",
            _view_module: "jupyter-exatomic",
            _model_name: "ExatomicSceneModel",
            _view_name: "ExatomicSceneView",
            clear: false,
            save: false,
            field_neg: "FF9900",
            field_pos: "003399",
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



var ExatomicSceneModel = widgets.DOMWidgetModel.extend({

    defaults: function() {
        return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_module_version: version,
            _view_module_version: version,
            _model_module: "jupyter-exatomic",
            _view_module: "jupyter-exatomic",
            _model_name: "ExatomicSceneModel",
            _view_name: "ExatomicSceneView",

        })
    }

});

var ExatomicSceneView = widgets.DOMWidgetView.extend({

    initialize: function() {
        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
        var that = this;
        $(this.el).width(
            this.model.get("layout").get("width")).height(
            this.model.get("layout").get("height")).resizable({
            aspectRatio: false,
            resize: function(event, ui) {
                event.preventDefault();
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
    },

    init: function() {
        this.app3d = new App3D(this);
        this.three_promises = this.app3d.init_promise();
        this.init_listeners();
        this.field_listeners();
    },

    resize: function(w, h) {
        this.model.get("layout").set("width", w);
        this.model.get("layout").set("height", h);
        this.app3d.resize();
    },

    render: function() {
        return this.app3d.finalize(this.three_promises);
    },


    save: function() {
        this.renderer.setSize(1920, 1080);
        this.camera.aspect = 1920 / 1080;
        this.camera.updateProjectionMatrix();
        this.render();
        var image = this.renderer.domElement.toDataURL("image/png");
        this.send({"type": "image", "content": image});
        var w = this.model.get("layout").get("width");
        var h = this.model.get("layout").get("height");
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.render();
    },

    get_fps: function() {
        var fps = {ox: this.model.get("field_ox"),
                   oy: this.model.get("field_oy"),
                   oz: this.model.get("field_oz"),
                   nx: this.model.get("field_nx"),
                   ny: this.model.get("field_ny"),
                   nz: this.model.get("field_nz"),
                   fx: this.model.get("field_fx"),
                   fy: this.model.get("field_fy"),
                   fz: this.model.get("field_fz")};
        fps["dx"] = (fps["fx"] - fps["ox"]) / (fps["nx"] - 1);
        fps["dy"] = (fps["fy"] - fps["oy"]) / (fps["ny"] - 1);
        fps["dz"] = (fps["fz"] - fps["oz"]) / (fps["nz"] - 1);
        return fps;
    },

    get_field_colors: function() {
        return {"pos": parseInt(this.model.get("field_pos"), 16),
                "neg": parseInt(this.model.get("field_neg"), 16)}
    },

    add_field: function() {},

    field_listeners: function() {
        this.listenTo(this.model, "change:field_nx", this.add_field);
        this.listenTo(this.model, "change:field_ny", this.add_field);
        this.listenTo(this.model, "change:field_nz", this.add_field);
        this.listenTo(this.model, "change:field_iso", this.add_field);
    },

    init_listeners: function() {
        this.listenTo(this.model, "change:clear", this.clear_meshes);
        this.listenTo(this.model, "change:save", this.save);
    },

});


module.exports = {
    ExatomicSceneModel: ExatomicSceneModel,
    ExatomicSceneView: ExatomicSceneView,
    ExatomicBoxModel: ExatomicBoxModel,
    ExatomicBoxView: ExatomicBoxView
}
