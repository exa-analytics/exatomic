// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=================
base.js
=================
JavaScript "frontend" complement of exatomic's Universe
for use within the Jupyter notebook interface.
*/

"use strict";
var widgets = require("@jupyter-widgets/base");
var control = require("@jupyter-widgets/controls");
var three = require("./appthree.js");
var utils = require("./utils.js");
var version = "~" + require("../package.json").version;

//// var datawidgets = require("jupyter-datawidgets");
//var ds = require("jupyter-dataserializers");
//
//// var deserialize = ds.array_serialization.deserialize;
//
//var FancySceneModel = widgets.DOMWidgetModel.extend({
//
//    defaults: function() {
//        return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
//            _model_module_version: version,
//            _view_module_version: version,
//            _model_module: "exatomic",
//            _view_module: "exatomic",
//            _model_name: "DataSceneModel",
//            _view_name: "DataSceneView"
//
//        });
//    }
//});
//
//
//var FancySceneView = widgets.DOMWidgetView.extend({
//
//    initialize: function() {
//        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
//        this.init();
//    },
//
//
//    init: function() {
//        var that = this;
//        this.displayed.then(function() {
//            that.app3d = new three.FancyApp(this);
//        })
//    },
//
//});
//
//
//
//var DataSceneModel = widgets.DOMWidgetModel.extend({
//
//    defaults: function() {
//        return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
//            _model_module_version: version,
//            _view_module_version: version,
//            _model_module: "exatomic",
//            _view_module: "exatomic",
//            _model_name: "DataSceneModel",
//            _view_name: "DataSceneView"
//
//        });
//    }
//// });
//},
//    {
//    serializers: _.extend({
//        a0: ds.array_serialization,
//    }, widgets.DOMWidgetModel.serializers)
//});
//
//var logerror = function(e) {console.log(e.message)};
//
//var de_array = function(obj, key) {
//    return Promise.resolve(dataserializers.array_serialization.deserialize(obj.model.get(key)))
//                  .then(p => {obj[key] = p}).catch(logerror)
//};
//
//
//var DataSceneView = widgets.DOMWidgetView.extend({
//
//    initialize: function() {
//        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
//        this.init();
//    },
//
//
//    init: function() {
//        var a = this.model.get("a0");
//        console.log("success");
//        // console.log(this.model.get("l0"));
//        // console.log(this.model.get("d0"));
//        // this.de_promises = Promise.all([
//        //     de_array(this, "array0")
//        // ]);
//        // console.log(this);
//    },
//
//    // render: function() {
//    //     // return Promise.resolve(this.de_promises);
//    // }
//
//});



var ExatomicBoxModel = control.BoxModel.extend({

    defaults: _.extend({}, control.BoxModel.prototype.defaults, {
            _model_module_version: version,
            _view_module_version: version,
            _model_module: "exatomic",
            _view_module: "exatomic",
            _model_name: "ExatomicBoxModel",
            _view_name: "ExatomicBoxView",
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
            // TODO :: Instead of referencing the first camera object
            //      :: just set camera.rotation (and camera.zoom??) to
            //      :: copy original camera.
            that.scene_ps = that.children_views.views[1].then(function(vbox) {
                var hboxs = vbox.children_views.views;
                var promises = Promise.all(hboxs).then(function(hbox) {
                    var subpromises = [];
                    for (var i = 0; i < hbox.length; i++) {
                        var scns = hbox[i].children_views.views;
                        for (var j = 0; j < scns.length; j++) {
                            subpromises.push(scns[j]);
                        };
                    };
                    return Promise.all(subpromises).then(p => p);
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
        var that = this;
        this.scene_ps.then(function(views) {
            if (that.model.get("linked")) {
                var controls = views[0].app3d.controls;
                var camera = views[0].app3d.camera;
                for (var i = 1; i < views.length; i++) {
                    var a = views[i].app3d;
                    a.camera = camera;
                    a.controls = a.init_controls();
                    a.controls.addEventListener("change", a.render.bind(a));
                    // views[i].el.removeEventListener('mousemove', a.mouseover_listener);
                };
            } else {
                var camera = views[0].app3d.camera;
                for (var i = 1; i < views.length; i++) {
                    var a = views[i].app3d;
                    a.camera = camera.clone();
                    a.controls = a.init_controls();
                    a.controls.addEventListener("change", a.render.bind(a));
                    // a.finalize_mouseover();
                };
            };
        });
    },

    init_listeners: function() {
        this.listenTo(this.model, "change:linked", this.link_controls);
    }

});


var ExatomicSceneModel = widgets.DOMWidgetModel.extend({

    defaults: function() {
        return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_module_version: version,
            _view_module_version: version,
            _model_module: "exatomic",
            _view_module: "exatomic",
            _model_name: "ExatomicSceneModel",
            _view_name: "ExatomicSceneView"

        })
    }

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
        if (this.model.get("test")) {
            if (this.model.get("uni")) { func = this.add_field }
            else { func = this.add_geometry };
            this.three_promises.then(func.bind(this))
                .then(this.app3d.set_camera.bind(this.app3d));
                //.then(this.app3d.set_camera_from_scene.bind(this.app3d));
        };
    },

    resize: function() {
        var w = this.el.offsetWidth;
        var h = this.el.offsetHeight - 5; // canvas is 5 smaller than div
        this.model.set("w", w);
        this.model.set("h", h);
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
            var field = this.model.get("field");
            var kind = this.model.get("field_kind");
            var ars = utils.gen_field_arrays(this.get_fps());
            var func = utils[field];
            if (field === "SolidHarmonic") {
                var fml = this.model.get("field_ml");
                var tf = func(ars, kind, fml);
                var name = "Sol.Har.," + kind + "," + fml;
            } else {
                var tf = func(ars, kind);
                var name = field + "," + kind;
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
        if (msg["type"] === "close") { this.app3d.close() };
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
        // this.listenTo(this.model, "change:h", this.resize);
        // Test container
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


// var ExatomicSceneModel = widgets.DOMWidgetModel.extend({
//
//     defaults: function() {
//         return _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
//             _model_module_version: version,
//             _view_module_version: version,
//             _model_module: "jupyter-exatomic",
//             _view_module: "jupyter-exatomic",
//             _model_name: "ExatomicSceneModel",
//             _view_name: "ExatomicSceneView",
//
//         })
//     }
//
// });
//
// var ExatomicSceneView = widgets.DOMWidgetView.extend({
//
//     initialize: function() {
//         console.log("ExatomicSceneView initialize");
//         widgets.DOMWidgetView.prototype.initialize.apply(this, arguments);
//         var that = this;
//         $(this.el).width(
//             this.model.get("layout").get("width")).height(
//             this.model.get("layout").get("height")).resizable({
//             aspectRatio: false,
//             resize: function(event, ui) {
//                 event.preventDefault();
//                 var w = ui.size.width;
//                 var h = ui.size.height;
//                 that.model.get("layout").set("width", w);
//                 that.model.get("layout").set("height", h);
//                 that.el.width = w;
//                 that.el.height = h;
//                 that.resize(w, h);
//             }
//         });
//         this.init();
//     },
//
//     init: function() {
//         console.log("ExatomicSceneView init");
//         this.app3d = new three.App3D(this);
//         this.three_promises = this.app3d.init_promise();
//         this.field_listeners();
//         this.init_listeners();
//     },
//
//     resize: function(w, h) {
//         this.model.get("layout").set("width", w);
//         this.model.get("layout").set("height", h);
//         this.app3d.resize();
//     },
//
//     render: function() {
//         return this.app3d.finalize(this.three_promises);
//     },
//
//     get_fps: function() {
//         var fps = {ox: this.model.get("field_ox"),
//                    oy: this.model.get("field_oy"),
//                    oz: this.model.get("field_oz"),
//                    nx: this.model.get("field_nx"),
//                    ny: this.model.get("field_ny"),
//                    nz: this.model.get("field_nz"),
//                    fx: this.model.get("field_fx"),
//                    fy: this.model.get("field_fy"),
//                    fz: this.model.get("field_fz")};
//         fps["dx"] = (fps["fx"] - fps["ox"]) / (fps["nx"] - 1);
//         fps["dy"] = (fps["fy"] - fps["oy"]) / (fps["ny"] - 1);
//         fps["dz"] = (fps["fz"] - fps["oz"]) / (fps["nz"] - 1);
//         return fps;
//     },
//
//     _handle_custom_msg: function(msg, clbk) {
//         if (msg["type"] === "close") { this.app3d.close() };
//         if (msg["type"] === "camera") {
//             this.app3d.set_camera_from_camera(msg["content"]);
//         };
//     },
//
//     clear_meshes: function() {
//         this.app3d.clear_meshes();
//     },
//
//     save: function() {
//         this.send({"type": "image", "content": this.app3d.save()});
//     },
//
//     save_camera: function() {
//         this.send({"type": "camera", "content": this.app3d.camera.toJSON()});
//     },
//
//     field_listeners: function() {
//         this.listenTo(this.model, "change:field_nx", this.add_field);
//         this.listenTo(this.model, "change:field_ny", this.add_field);
//         this.listenTo(this.model, "change:field_nz", this.add_field);
//         this.listenTo(this.model, "change:field_iso", this.add_field);
//     },
//
//     init_listeners: function() {
//         this.listenTo(this.model, "change:clear", this.clear_meshes);
//         this.listenTo(this.model, "change:save", this.save);
//         this.listenTo(this.model, "change:save_cam", this.save_camera);
//         this.listenTo(this.model, "msg:custom", this._handle_custom_msg);
//     },
//
// });


module.exports = {
    ExatomicSceneModel: ExatomicSceneModel,
    ExatomicSceneView: ExatomicSceneView,
//    DataSceneModel: DataSceneModel,
//    DataSceneView: DataSceneView,
    ExatomicBoxModel: ExatomicBoxModel,
    ExatomicBoxView: ExatomicBoxView,
    unpack_models: widgets.unpack_models
}
