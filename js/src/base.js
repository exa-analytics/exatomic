// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=================
base.js
=================
JavaScript "frontend" complement of exatomic"s Universe
for use within the Jupyter notebook interface.
*/

"use strict";
var widgets = require("@jupyter-widgets/base");
var control = require("@jupyter-widgets/controls");
var _ = require("underscore");
var three = require("./appthree");
var utils = require("./utils");
var semver = "^" + require("../package.json").version;
console.log("exatomic JS version: " + require("../package.json").version);


var ExatomicBoxModel = control.BoxModel.extend({

    defaults: _.extend({}, control.BoxModel.prototype.defaults, {
            _model_name: "ExatomicBoxModel",
            _view_name: "ExatomicBoxView",
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: "exatomic",
            _view_module: "exatomic",
            linked: false
    })

});


var ExatomicBoxView = control.BoxView.extend({

    initialize: function() {
        control.BoxView.prototype.initialize.apply(this, arguments);
        this.init();
    },

    init: function() {
        this.init_listeners();
        var that = this;
        this.displayed.then(function() {
            that.scene_ps = that.children_views.views[1].then(function(vbox) {
                var hboxs = vbox.children_views.views;
                var promises = Promise.all(hboxs).then(function(hbox) {
                    var subpromises = [];
                    for (var i = 0; i < hbox.length; i++) {
                        var scns = hbox[i].children_views.views;
                        for (var j = 0; j < scns.length; j++) {
                            subpromises.push(scns[j]);
                        };
                    }
                    return Promise.all(subpromises).then((p) => p);
                });
                return promises;
            });
            that.scene_ps.then(function(p) {
                for (var i = 0; i < p.length; i++) {
                    p[i].resize();
                }
            })
        });
    },

    link_controls: function() {
        // TODO :: Instead of referencing the first camera object
        //      :: just set camera.rotation (and camera.zoom??) to
        //      :: copy original camera.
        //      :: e.g. -- camera[i].rotation.copy(camera[0])
        var i, app;
        var that = this;
        this.scene_ps.then(function(views) {
            if (that.model.get("linked")) {
                var idxs = that.model.get("active_scene_indices");
                var controls = views[idxs[0]].app3d.controls;
                var camera = views[idxs[0]].app3d.camera;
                for (i = 1; i < idxs.length; i++) {
                    app = views[idxs[i]].app3d;
                    app.camera = camera;
                    app.controls = app.init_controls();
                    app.controls.addEventListener("change", app.render.bind(app));
                };
            } else {
                for (i = 0; i < views.length; i++) {
                    app = views[i].app3d;
                    app.camera = app.camera.clone();
                    app.controls = app.init_controls();
                    app.controls.addEventListener("change", app.render.bind(app));
                };
            };
        });
    },

    init_listeners: function() {
        this.listenTo(this.model, "change:linked", this.link_controls);
    }

});


var ExatomicSceneModel = widgets.DOMWidgetModel.extend({

    defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
        _model_name: "ExatomicSceneModel",
        _view_name: "ExatomicSceneView",
        _model_module_version: semver,
        _view_module_version: semver,
        _model_module: "exatomic",
        _view_module: "exatomic"
    })

});


var ExatomicSceneView = widgets.DOMWidgetView.extend({

    initialize: function() {
        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
        this.init_listeners();
        this.init();
    },

    init: function() {
        var func;
        window.addEventListener("resize", this.resize.bind(this));
        this.app3d = new three.App3D(this);
        this.three_promises = this.app3d.init_promise();
        if (this.model.get("uni")) {
            func = this.add_field;
        } else {
            func = this.add_geometry;
        };
        this.three_promises.then(func.bind(this))
            .then(this.app3d.set_camera.bind(this.app3d));
    },

    resize: function() {
        // sometimes during window resize these are 0
        var w = this.el.offsetWidth || 200;
        var h = this.el.offsetHeight || 200;
        this.model.set("w", w);
        // threejs canvas is 5 smaller than div
        this.model.set("h", h - 5);
    },

    render: function() {
        return this.app3d.finalize(this.three_promises);
    },

    add_geometry: function(color) {
        this.app3d.clear_meshes("generic");
        if (this.model.get("geom")) {
            this.app3d.meshes["generic"] = this.app3d.test_mesh();
            this.app3d.add_meshes("generic");
        };
    },

    colors: function() {
        return {"pos": this.model.get("field_pos"),
                "neg": this.model.get("field_neg")};
    },

    add_field: function() {
        this.app3d.clear_meshes("field");
        if (this.model.get("uni")) {
            var name, tf;
            var field = this.model.get("field");
            var kind = this.model.get("field_kind");
            var ars = utils.gen_field_arrays(this.get_fps());
            var func = utils[field];
            if (field === "SolidHarmonic") {
                var fml = this.model.get("field_ml");
                tf = func(ars, kind, fml);
                name = "Sol.Har.," + kind + "," + fml;
            } else {
                tf = func(ars, kind);
                name = field + "," + kind;
            };
            this.app3d.meshes["field"] = this.app3d.add_scalar_field(
                tf, this.model.get("field_iso"),
                this.model.get("field_o"), 2,
                this.colors());
            for (var i = 0; i < this.app3d.meshes["field"].length; i++) {
                this.app3d.meshes["field"][i].name = name;
            };
        } else {
            this.app3d.meshes["field"] = this.app3d.add_scalar_field(
                utils.scalar_field(
                    utils.gen_field_arrays(this.get_fps()),
                    utils[this.model.get("field")]
                ),
                this.model.get("field_iso"),
                this.model.get("field_o")
            );
            this.app3d.meshes["field"][0].name = this.model.get("field");
        };
        this.app3d.add_meshes("field");
    },

    update_field: function() {
        var meshes = this.app3d.meshes["field"];
        for (var i = 0; i < meshes.length; i++) {
            meshes[i].material.transparent = true;
            meshes[i].material.opacity = this.model.get("field_o");
            meshes[i].material.needsUpdate = true;
        };
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

    clear_meshes: function() {
        this.app3d.clear_meshes();
    },


    save: function() {
        this.send({"type": "image", "content": this.app3d.save()});
    },

    save_camera: function() {
        this.send({"type": "camera", "content": this.app3d.camera.toJSON()});
    },

    _handle_custom_msg: function(msg, clbk) {
        if (msg["type"] === "close") { this.app3d.close(); clearInterval(this.interv); };
        if (msg["type"] === "camera") {
            this.app3d.set_camera_from_camera(msg["content"]);
        };
    },

    init_listeners: function() {
        // The basics
        this.listenTo(this.model, "change:clear", this.clear_meshes);
        this.listenTo(this.model, "change:save", this.save);
        this.listenTo(this.model, "change:save_cam", this.save_camera);
        this.listenTo(this.model, "msg:custom", this._handle_custom_msg);
        this.listenTo(this.model, "change:geom", this.add_geometry);
        // Field stuff
        if (!this.model.get("uni")) {
            this.listenTo(this.model, "change:field", this.add_field);
        };
        this.listenTo(this.model, "change:field_kind", this.add_field);
        this.listenTo(this.model, "change:field_ml", this.add_field);
        this.listenTo(this.model, "change:field_o", this.update_field);
        this.listenTo(this.model, "change:field_nx", this.add_field);
        this.listenTo(this.model, "change:field_ny", this.add_field);
        this.listenTo(this.model, "change:field_nz", this.add_field);
        this.listenTo(this.model, "change:field_iso", this.add_field);
    },

});


module.exports = {
    ExatomicSceneModel: ExatomicSceneModel,
    ExatomicSceneView: ExatomicSceneView,
    ExatomicBoxModel: ExatomicBoxModel,
    ExatomicBoxView: ExatomicBoxView
}

ExatomicSceneView.prototype.send_obj = function() {
    var _this = this;
    _this.interv = setInterval(function() {
        //_this.send({"type": "object", "content": _this.app3d.probe()});
        var vals = _this.app3d.probe();
        _this.model.set("obj", vals[0]);
        console.log(vals);
    }, 1000);
};
