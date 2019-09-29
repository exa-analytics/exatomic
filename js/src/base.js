// Copright (c) 2015-2018, Exa Analytics Development Team)
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
var _ = require("underscore");
var three = require("./appthree");
var utils = require("./utils");
// var myapp = require("./app");
var semver = "^" + require("../package.json").version;


console.log("exatomic JS version: " + require("../package.json").version);
console.log("trogdor burninating the countryside")


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


var ThreeSceneModel = widgets.DOMWidgetModel.extend({

    defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
        _model_name: "ThreeSceneModel",
        _view_name: "ThreeSceneView",
        _model_module_version: semver,
        _view_module_version: semver,
        _model_module: "exatomic",
        _view_module: "exatomic"
    })

});

var ThreeSceneView = widgets.DOMWidgetView.extend({

    initialize: function() {
        widgets.DOMWidgetView.prototype.initialize.apply(this, arguments)
        this.init()
        this.init_listeners()
    },

    render: function() {
        return this.promises
    },

    parse: function(key) {
        return Promise.resolve(
            JSON.parse(this.model.get(key))
        ).then((arr) => {
            if (arr.length) {
                console.log("parsed json", key, arr.length, "frames")
            }
            this[key] = arr
        })
    },

    setattr: function(key) {
        return Promise.resolve(
            this.model.get(key)
        ).then((obj) => {
            if (Object.keys(obj).length) {
                console.log("setting attr", key)
            }
            this[key] = obj
        })
    },

    init: function() {
        this.app3d = new three.NewApp3D(this)
       // this.test_app = new myapp.App(this)
        this.promises = Promise.all([
         //   this.test_app.init(),
            this.app3d.init(),
            this.setattr("atom_x"),
            this.setattr("atom_y"),
            this.setattr("atom_z"),
            this.setattr("atom_s"),
            this.setattr("field_val")
        ])
        this.promises.then(() => {
            this.set_test()
            this.set_dims()
            this.set_atom()
            this.app3d.set_camera_from_scene()
        })
    },

    set_atom_x: function() { this.setattr("atom_x") },
    set_atom_y: function() { this.setattr("atom_y") },
    set_atom_z: function() { this.setattr("atom_z") },
    set_atom_s: function() { this.setattr("atom_s") },

    init_listeners: function() {
        this.listenTo(this.model, "change:dims", this.set_dims)
        this.listenTo(this.model, "change:atom", this.set_atom)
        this.listenTo(this.model, "change:test", this.set_test)
        this.listenTo(this.model, "change:frame", this.set_atom)
        this.listenTo(this.model, "change:field", this.set_field)
        this.listenTo(this.model, "change:filled", this.set_atom)
        this.listenTo(this.model, "change:atom_x", this.set_atom_x)
        this.listenTo(this.model, "change:atom_y", this.set_atom_y)
        this.listenTo(this.model, "change:atom_z", this.set_atom_z)
        this.listenTo(this.model, "change:atom_s", this.set_atom_s)
        this.listenTo(this.model, "change:field_alp", this.update_field)
        this.listenTo(this.model, "change:field_idx", this.set_field)
        this.listenTo(this.model, "change:field_iso", this.set_field)
        this.listenTo(this.model, "change:field_fun", this.set_field)
        this.listenTo(this.model, "change:field_typ", this.set_field)
        this.listenTo(this.model, "change:field_sub", this.set_field)
        this.listenTo(this.model, "change:recording", this.set_recording)
        this.listenTo(this.model, "change:camera_origin", this.set_camera_origin)
        this.listenTo(this.model, "change:camera_position", this.set_camera_position)
    },

    get_fps: function() {
        return {
            "ox": this.model.get("field_ox"),
            "oy": this.model.get("field_oy"),
            "oz": this.model.get("field_oz"),
            "nx": this.model.get("field_nx"),
            "ny": this.model.get("field_ny"),
            "nz": this.model.get("field_nz"),
            "fx": this.model.get("field_fx"),
            "fy": this.model.get("field_fy"),
            "fz": this.model.get("field_fz")
        }
    },

    get_colors: function() {
        return {
            "pos": this.model.get("field_pos"),
            "neg": this.model.get("field_neg")
        }
    },

    set_recording: function() {
        this.app3d.recording = this.model.get("recording")
    },

    update_field: function() {
        var meshes = this.app3d.meshes["field"]
        for (var i = 0; i < meshes.length; i++) {
            meshes[i].material.transparent = true
            meshes[i].material.opacity = this.model.get("field_alp")
            meshes[i].material.needsUpdate = true
        }
    },

    _handle_custom_msg: function(msg, clbk) {
        if (msg["type"] === "close") {
            this.app3d.close()
        }
        if (msg["type"] === "camera") {
            this.app3d.set_camera_from_camera(msg["content"])
        }
    },

    set_atom: function() {
        this.app3d.clear_meshes("atom")
        let atom = this.model.get("atom")
        console.log("setting atom", atom)
        if (atom) {
            // this.app3d.clear_meshes("test")
            let func
            let frame = this.model.get("frame")
            if (typeof this.atom_x[frame] === "string") {
                this.atom_x[frame] = JSON.parse(this.atom_x[frame])
                this.atom_y[frame] = JSON.parse(this.atom_y[frame])
                this.atom_z[frame] = JSON.parse(this.atom_z[frame])
                this.atom_s[frame] = JSON.parse(this.atom_s[frame])
            }
            console.log(this.atom_x[frame])
            if (this.atom_x[frame].length) {
                if (this.model.get("filled")) {
                    func = this.app3d.add_spheres
                    console.log("drawing spheres for frame", frame)
                } else {
                    func = this.app3d.add_points
                    console.log("drawing points for frame", frame)
                }
                this.app3d.meshes["atom"] = func(
                    this.atom_x[frame],
                    this.atom_y[frame],
                    this.atom_z[frame],
                    this.atom_s[frame].map(s => this.model.get("atom_c")[s]),
                    this.atom_s[frame].map(s => this.model.get("atom_cr")[s]),
                    this.atom_s[frame].map(s => this.model.get("atom_l")[s]))
                this.app3d.add_meshes("atom")
            }
        }
    },

    set_field: function() {
        this.app3d.clear_meshes("field")
        let field = this.model.get("field")
        console.log("setting field", field)
        if (field) {
            let fld, fun
            let idx = this.model.get("field_idx")
            let val = this.model.get("field_val")
            let nam = this.model.get("field_fun")
            let iso = this.model.get("field_iso")
            let alp = this.model.get("field_alp")
            let col = this.get_colors()
            // First check for fields passed from python
            if (val.length) {
                fld = this.field_val[idx]
                if (typeof fld === "string") {
                    console.log("parsing field array from python")
                    this.field_val[idx] = JSON.parse(fld)
                }
                fld = utils.scalar_field(this.get_fps(), fld)
                console.log("adding field from array")
                this.app3d.add_scalar_field(fld, iso, alp, 2, col)
                let lab = this.model.get("field_lab")[idx]
                this.app3d.meshes["field"][0].name = `${lab}(${iso})`
                this.app3d.meshes["field"][1].name = `${lab}(-${iso})`
            // Otherwise generate fields from func in
            //     Torus, Sphere, Ellipsoid,
            //     Gaussian, Hydrogenic, SolidHarmonic
            } else if (nam) {
                if (["Torus", "Sphere", "Ellipsoid"].includes(nam)) {
                    console.log("adding plain func", nam)
                    fun = utils[nam]
                    fld = utils.scalar_field(this.get_fps(), fun)
                    this.app3d.add_scalar_field(fld, iso, alp)
                    this.app3d.meshes["field"][0].name = `${nam}(${iso})`
                } else {
                    console.log("adding configurable func")
                    console.log("typ", "sub",
                                this.model.get("field_typ"),
                                this.model.get("field_sub"))
                    fun = utils[nam](this.model.get("field_typ"),
                                     this.model.get("field_sub"))
                    fld = utils.scalar_field(this.get_fps(), fun)
                    this.app3d.add_scalar_field(fld, iso, alp, 2, this.get_colors())
                    this.app3d.meshes["field"][0].name =`${nam}(${iso})`
                    this.app3d.meshes["field"][1].name =`${nam}(-${iso})`
                }
            }
        }
    },

    set_dims: function() {
        let dims = this.model.get("dims")
        console.log("setting scene dims", dims)
        let w = dims[0]
        let h = dims[1]
        this.app3d.w = w
        this.app3d.h = h
        this.app3d.w2 = w / 2
        this.app3d.h2 = h / 2
        this.app3d.camera.aspect = w / h
        this.app3d.renderer.setSize(w, h)
        this.app3d.gpicker.resizeTexture(w, h)
        this.app3d.hudcamera.left   = -w / 2
        this.app3d.hudcamera.right  =  w / 2
        this.app3d.hudcamera.top    =  h / 2
        this.app3d.hudcamera.bottom = -h / 2
    },

    set_test: function() {
        let test = this.model.get("test")
        console.log("setting test", test)
        this.app3d.test_mesh(test)
    },

    set_camera_position: function() {
        let pos = this.model.get("camera_position")
        console.log("setting camera position", pos)
        this.app3d.camera.position.set(pos[0], pos[1], pos[2])
    },

    set_camera_origin: function() {
        let orig = this.model.get("camera_origin")
        console.log("setting camera origin", orig)
        this.app3d.set_camera_origin(orig[0], orig[1], orig[2])
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
    ExatomicBoxView: ExatomicBoxView,
    ThreeSceneModel: ThreeSceneModel,
    ThreeSceneView: ThreeSceneView
}

