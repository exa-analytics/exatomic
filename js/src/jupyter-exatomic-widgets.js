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
var base = require("./jupyter-exatomic-base.js");
var utils = require("./jupyter-exatomic-utils.js");


var UniverseSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "UniverseSceneModel",
            _view_name: "UniverseSceneView",
            frame_idx: 0,
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
        // TODO :: Set up Promises so that initialization
        //         does not make the browser hang forever.
        UniverseSceneView.__super__.init.apply(this);
        this.app3d.set_camera({"x": 40.0, "y": 40.0, "z": 40.0});
        this.frame_idx = this.model.get("frame_idx");
        // Atom
        this.atom_x = JSON.parse(this.model.get("atom_x"));
        this.atom_y = JSON.parse(this.model.get("atom_y"));
        this.atom_z = JSON.parse(this.model.get("atom_z"));
        this.atom_s = JSON.parse(this.model.get("atom_s"));
        this.atom_r = this.model.get("atom_r");
        this.atom_c = this.model.get("atom_c");
        // Two
        if (this.model.get("two_b0") !== null) {
            this.two_b0 = JSON.parse(this.model.get("two_b0"));
            this.two_b1 = JSON.parse(this.model.get("two_b1"));
        };
        // Field
        if (this.model.get("field_i") !== null) {
            this.field_i = this.model.get("field_i");
            this.field_v = this.model.get("field_v");
            this.field_p = this.model.get("field_p");
        };
        // Frame
        if (this.model.get("frame") !== null) {

        };
        this.init_listeners();
        this.add_atom();
        this.animation();
    },

    add_atom: function() {
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
        if (this.two_b0 !== undefined) {
            this.meshes["two"] = bond(this.two_b0[fdx], this.two_b1[fdx],
                                      this.atom_x[fdx], this.atom_y[fdx],
                                      this.atom_z[fdx], colrs);
        };
        this.add_meshes();
    },

    add_field: function() {
        this.clear_meshes("field");
        var fldx = this.model.get("field_idx");
        if (fldx === 'null') { return };
        var iso = this.model.get("field_iso");
        var fdx = this.model.get("frame_idx");
        var idx = this.field_i[fdx][fldx];
        var fps = this.field_p[fdx][fldx];
        var ars = utils.gen_field_arrays(fps);
        var vals = this.field_v[idx];
        var tf = utils.scalar_field(ars, vals);
        this.meshes["field"] = this.app3d.add_scalar_field(tf, iso, 2);
        this.add_meshes("field");
    },

    init_listeners: function() {
        UniverseSceneView.__super__.init_listeners.apply(this);
        this.listenTo(this.model, "change:frame_idx", this.add_atom);
        this.listenTo(this.model, "change:atom_3d", this.add_atom);
        this.listenTo(this.model, "change:field_idx", this.add_field);
    }

});

var UniverseWidgetModel = base.ExatomicBoxModel.extend({
    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "UniverseWidgetModel",
        _view_name: "UniverseWidgetView",
    })
});

var UniverseWidgetView = base.ExatomicBoxView.extend({});



module.exports = {
    UniverseWidgetModel: UniverseWidgetModel,
    UniverseWidgetView: UniverseWidgetView,
    UniverseSceneModel: UniverseSceneModel,
    UniverseSceneView: UniverseSceneView
}

