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
var base = require("./jupyter-exatomic-base.js");
var utils = require("./jupyter-exatomic-utils.js");


var UniverseSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "UniverseSceneModel",
            _view_name: "UniverseSceneView",
            frame_idx: 0,
            field_show: false,
            cont_show: false,
            cont_axis: "z",
            cont_num: 10,
            cont_lim: [-8, -1],
            field_idx: "null",
            field_iso: 0.03,
            field_i: "",
            field_v: "",
            field_p: {},
            atom_x: "",
            atom_y: "",
            atom_z: "",
            atom_s: "",
            atom_r: {},
            arom_c: {},
            two_b0: "",
            two_b1: "",
        })
    }

});


var UniverseSceneView = base.ExatomicSceneView.extend({

    init: function() {
        // TODO :: Fix Promises so that initialization
        //         accounts for THREE.js initialization
        //         and allows for dynamically populating
        //         the initial scene (atom frame 0).
        base.ExatomicSceneView.prototype.init.apply(this);
        this.app3d.set_camera({"x": 40.0, "y": 40.0, "z": 40.0});
        // Atom
        var that = this;
        var jsonparse = function(string) {
            return new Promise(function(resolve, reject) {
                try {
                    var obj = JSON.parse(string);
                    resolve(obj);
                } catch(err) {
                    reject(err);
                }
            });
        };
        var promises = [
            jsonparse(this.model.get("atom_x"))
                .then(function(p) {that.atom_x = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("atom_y"))
                .then(function(p) {that.atom_y = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("atom_z"))
                .then(function(p) {that.atom_z = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("atom_s"))
                .then(function(p) {that.atom_s = p})
                .catch(function(e) {console.log(e.message)}),
            Promise.resolve(this.model.get("atom_r"))
                .then(function(p) {that.atom_r = p})
                .catch(function(e) {console.log(e.message)}),
            Promise.resolve(this.model.get("atom_c"))
                .then(function(p) {that.atom_c = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("two_b0"))
                .then(function(p) {that.two_b0 = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("two_b1"))
                .then(function(p) {that.two_b1 = p})
                .catch(function(e) {console.log(e.message)}),
            Promise.resolve(this.model.get("field_i"))
                .then(function(p) {that.field_i = p})
                .catch(function(e) {console.log(e.message)}),
            Promise.resolve(this.model.get("field_p"))
                .then(function(p) {that.field_p = p})
                .catch(function(e) {console.log(e.message)}),
            jsonparse(this.model.get("field_v"))
                .then(function(p) {that.field_v = p})
                .catch(function(e) {console.log(e.message)})
        ];
        this.promises = Promise.all(promises); //.then(that.add_atom);
    },

    add_atom: function() {
        console.log(this);
        this.clear_meshes("atom");
        this.clear_meshes("two");
        var fdx = this.model.get("frame_idx");
        var syms = this.atom_s[fdx];
        var colrs = utils.mapper(syms, this.atom_c);
        var radii = utils.mapper(syms, this.atom_r);
        if (this.model.get("atom_3d")) {
            var atom = this.app3d.add_spheres;
            var bond = this.app3d.add_cylinders;
        } else {
            var atom = this.app3d.add_points;
            var bond = this.app3d.add_lines;
        };
        this.meshes["atom"] = atom(this.atom_x[fdx], this.atom_y[fdx],
                                   this.atom_z[fdx], colrs, radii);
        if (this.two_b0.length !== 0) {
            this.meshes["two"] = bond(this.two_b0[fdx], this.two_b1[fdx],
                                      this.atom_x[fdx], this.atom_y[fdx],
                                      this.atom_z[fdx], colrs);
        };
        this.add_meshes();
    },

    add_field: function() {
        this.clear_meshes("field");
        if (this.model.get("field_show") === false) { return };
        var fldx = this.model.get("field_idx");
        if (fldx === "null") { return };
        var fdx = this.model.get("frame_idx");
        var idx = this.field_i[fdx][fldx];
        var fps = this.field_p[fdx][fldx];
        this.meshes["field"] = this.app3d.add_scalar_field(
            utils.scalar_field(
                utils.gen_field_arrays(fps),
                this.field_v[idx]
            ),
            this.model.get("field_iso"),
            2, this.get_field_colors()
        );
        this.add_meshes("field");
    },

    add_contour: function() {
        this.clear_meshes("contour");
        if (this.model.get("cont_show") === false) { return };
        var fldx = this.model.get("field_idx");
        if (fldx === "null") { return };
        var fdx = this.model.get("frame_idx");
        var idx = this.field_i[fdx][fldx];
        var fps = this.field_p[fdx][fldx];
        this.meshes["contour"] = this.app3d.add_contour(
            utils.scalar_field(
                utils.gen_field_arrays(fps),
                this.field_v[idx]
            ),
            this.model.get("cont_num"),
            this.model.get("cont_lim"),
            this.model.get("cont_axis"),
            this.model.get("cont_val"),
            this.get_field_colors()
        );
        this.add_meshes("contour");
    },

    init_listeners: function() {
        base.ExatomicSceneView.prototype.init_listeners.apply(this);
        this.listenTo(this.model, "change:frame_idx", this.add_atom);
        this.listenTo(this.model, "change:atom_3d", this.add_atom);
        this.listenTo(this.model, "change:field_idx", this.add_field);
        this.listenTo(this.model, "change:field_show", this.add_field);
        this.listenTo(this.model, "change:field_idx", this.add_contour);
        this.listenTo(this.model, "change:cont_show", this.add_contour);
        this.listenTo(this.model, "change:cont_axis", this.add_contour);
        this.listenTo(this.model, "change:cont_num", this.add_contour);
        this.listenTo(this.model, "change:cont_lim", this.add_contour);
        this.listenTo(this.model, "change:cont_val", this.add_contour);
    }

});


var UniverseWidgetModel = base.ExatomicBoxModel.extend({

    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "UniverseWidgetModel",
        _view_name: "UniverseWidgetView",
    })

});


var UniverseWidgetView = base.ExatomicBoxView.extend({

});


module.exports = {
    UniverseWidgetModel: UniverseWidgetModel,
    UniverseWidgetView: UniverseWidgetView,
    UniverseSceneModel: UniverseSceneModel,
    UniverseSceneView: UniverseSceneView
}
