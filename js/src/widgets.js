// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=================
widgets.js
=================
JavaScript "frontend" complement of exatomic's Container for use within
the Jupyter notebook interface.
*/

"use strict";
var base = require("./base");
var utils = require("./utils");
var three = require("./appthree");
var _ = require('underscore');


var UniverseSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "UniverseSceneModel",
            _view_name: "UniverseSceneView"
        })
    }

});


var UniverseSceneView = base.ExatomicSceneView.extend({

    init: function() {
        window.addEventListener("resize", this.resize.bind(this));
        this.app3d = new three.App3D(this);
        this.three_promises = this.app3d.init_promise();
        this.promises = Promise.all([utils.fparse(this, "atom_x"),
            utils.fparse(this, "atom_y"), utils.fparse(this, "atom_z"),
            utils.fparse(this, "atom_s"), utils.mesolv(this, "atom_r"),
            utils.mesolv(this, "atom_c"), utils.fparse(this, "atom_l"),
            utils.fparse(this, "two_b0"), utils.fparse(this, "two_b1"),
            utils.mesolv(this, "field_i"), utils.mesolv(this, "field_p"),
            utils.mesolv(this, "field_v")]);
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.add_atom.bind(this))
            .then(this.generate_tensor.bind(this))
            .then(this.app3d.set_camera_from_scene.bind(this.app3d));
    },

    render: function() {
        return Promise.all([this.three_promises, this.promises]);
    },

    add_atom: function() {
        this.app3d.clear_meshes("atom");
        this.app3d.clear_meshes("two");
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
        var labels = (this.atom_l) ? this.atom_l[fdx] : "";
        this.app3d.meshes["atom"] = atom(
            this.atom_x[fdx], this.atom_y[fdx],
            this.atom_z[fdx], colrs, radii, labels);
        if (this.two_b0.length !== 0) {
            this.app3d.meshes["two"] = bond(
                this.two_b0[fdx], this.two_b1[fdx],
                this.atom_x[fdx], this.atom_y[fdx],
                this.atom_z[fdx], colrs);
        };
        this.app3d.add_meshes();
    },

    // parse_or_return_field: function(idx) {
    //     // var fldx = this.model.get("field_idx");
    //     // if (fldx === "null") { return };
    //     // var fdx = this.model.get("frame_idx");
    //     // var idx = this.field_i[fdx][fldx];
    //     if (typeof this.field_v[idx] === 'string') {
    //         utils.jsonparse(this.field_v[idx]).then(f => f);
    //     };
    //     return this.field_v[idx];
    // },

    add_field: function() {
        this.app3d.clear_meshes("field");
        if (!this.model.get("field_show")) { return };
        var fldx = this.model.get("field_idx");
        if (fldx === "null") { return };
        var fdx = this.model.get("frame_idx");
        // var values = JSON.Parse(field_v[idx]);
        var fps = this.field_p[fdx][fldx];
        if (fps === undefined) { return };
        var idx = this.field_i[fdx][fldx];
        var that = this;
        if (typeof this.field_v[idx] === 'string') {
            utils.jsonparse(this.field_v[idx])
                .then(function(values) {
                    that.field_v[idx] = values;
                    that.app3d.meshes["field"] = that.app3d.add_scalar_field(
                        utils.scalar_field(
                            utils.gen_field_arrays(fps),
                            values),
                    that.model.get("field_iso"),
                    that.model.get("field_o"), 2,
                    that.colors());
                    that.app3d.add_meshes("field");
                });
        } else {
            this.app3d.meshes["field"] = this.app3d.add_scalar_field(
                utils.scalar_field(
                    utils.gen_field_arrays(fps),
                    this.field_v[idx]),
                this.model.get("field_iso"),
                this.model.get("field_o"), 2,
                this.colors());
            this.app3d.add_meshes("field");
        };
    },

    add_contour: function() {
        this.app3d.clear_meshes("contour");
        if (!this.model.get("cont_show")) { return };
        var fldx = this.model.get("field_idx");
        // Specifically test for string null
        if (fldx === "null") { return };
        var fdx = this.model.get("frame_idx");
        var idx = this.field_i[fdx][fldx];
        var fps = this.field_p[fdx][fldx];
        if (fps === undefined) { return };
        var that = this;
        if (typeof this.field_v[idx] === 'string') {
            utils.jsonparse(this.field_v[idx])
                .then(function(values) {
                    that.field_v[idx] = values;
                    that.app3d.meshes["contour"] = that.app3d.add_contour(
                        utils.scalar_field(
                            utils.gen_field_arrays(fps),
                            values),
                        that.model.get("cont_num"),
                        that.model.get("cont_lim"),
                        that.model.get("cont_axis"),
                        that.model.get("cont_val"),
                        that.colors());
                    that.app3d.add_meshes("contour");
                });
        } else {
            this.app3d.meshes["contour"] = this.app3d.add_contour(
                utils.scalar_field(
                    utils.gen_field_arrays(fps),
                    this.field_v[idx]),
                this.model.get("cont_num"),
                this.model.get("cont_lim"),
                this.model.get("cont_axis"),
                this.model.get("cont_val"),
                this.colors());
            that.app3d.add_meshes("contour");
        };
    },

    add_axis: function() {
        this.app3d.clear_meshes("generic");
        if (this.model.get("axis")) {
            this.app3d.meshes["generic"] = this.app3d.add_unit_axis(
                this.model.get("atom_3d"));
        };
        this.app3d.add_meshes("generic");
    },

    get_tensor: function() {
        return [
            [this.model.get("txx"), this.model.get("txy"), this.model.get("txz")],
            [this.model.get("tyx"), this.model.get("tyy"), this.model.get("tyz")],
            [this.model.get("tzx"), this.model.get("tzy"), this.model.get("tzz")]]
    },

    generate_tensor: function() {
        this.app3d.clear_meshes("generic");
        var atomIndex = this.model.get("tensorAtom");
        var sceneIndex = this.model.get("activeTensor");
        var scaling = this.model.get("scale");
        //console.log(sceneIndex);
        if ( this.model.get("tens") ) {
            //console.log("hallo");
            this.app3d.meshes["generic"] = 
                    this.app3d.add_tensor_surface( this.get_tensor(),
                        this.atom_x[sceneIndex][atomIndex],
                        this.atom_y[sceneIndex][atomIndex],
                        this.atom_z[sceneIndex][atomIndex],
                        scaling );
            //console.log("generic");
        };
        this.app3d.add_meshes("generic");
    },

    init_listeners: function() {
        base.ExatomicSceneView.prototype.init_listeners.call(this);
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
        this.listenTo(this.model, "change:atom_3d", this.add_axis);
        this.listenTo(this.model, "change:axis", this.add_axis);

        //Tensor listeners
        this.listenTo(this.model, "change:tens", this.generate_tensor);
        this.listenTo(this.model, "change:txx", this.generate_tensor);
        this.listenTo(this.model, "change:txy", this.generate_tensor);
        this.listenTo(this.model, "change:txz", this.generate_tensor);
        this.listenTo(this.model, "change:tyx", this.generate_tensor);
        this.listenTo(this.model, "change:tyy", this.generate_tensor);
        this.listenTo(this.model, "change:tyz", this.generate_tensor);
        this.listenTo(this.model, "change:tzx", this.generate_tensor);
        this.listenTo(this.model, "change:tzy", this.generate_tensor);
        this.listenTo(this.model, "change:tzz", this.generate_tensor);
        this.listenTo(this.model, "change:scale", this.generate_tensor);
        this.listenTo(this.model, "change:tensorAtom", this.generate_tensor);
        this.listenTo(this.model, "change:activeTensor", this.generate_tensor);
    }

});


module.exports = {
    UniverseSceneModel: UniverseSceneModel,
    UniverseSceneView: UniverseSceneView
}
