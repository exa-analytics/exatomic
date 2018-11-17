// Copright (c) 2015-2018, Exa Analytics Development Team
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
        });
    }

});


var UniverseSceneView = base.ExatomicSceneView.extend({

    init: function() {
        window.addEventListener("resize", this.resize.bind(this));
        this.app3d = new three.App3D(this);
        this.three_promises = this.app3d.init_promise();
        this.promises = Promise.all([utils.fparse(this, "atom_x"),
            utils.fparse(this, "atom_y"), utils.fparse(this, "atom_z"),
            utils.fparse(this, "atom_s"), utils.mesolv(this, "atom_cr"),
            utils.mesolv(this, "atom_vr"),
            utils.mesolv(this, "atom_c"), utils.mesolv(this, "atom_l"),
            utils.fparse(this, "two_b0"), utils.fparse(this, "two_b1"),
            utils.mesolv(this, "field_i"), utils.mesolv(this, "field_p"),
            utils.mesolv(this, "field_v"), utils.mesolv(this, "tensor_d")]);
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.add_atom.bind(this))
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
        var atom, bond, radii, r = ((this.model.get("bond_r") > 0) ? this.model.get("bond_r") : 0.15);
        if (this.model.get("atom_3d")) {
            switch (this.model.get("fill_idx")) {
                case 0:
                    radii = utils.mapper(syms, this.atom_cr)
                                    .map(function(x) { return x * 0.5; });
                    atom = this.app3d.add_spheres;
                    bond = this.app3d.add_cylinders;
                    break;
                case 1:
                    radii = utils.mapper(syms, this.atom_vr);
                    atom = this.app3d.add_spheres;
                    bond = null;
                    break;
                case 2:
                    radii = utils.mapper(syms, this.atom_cr);
                    atom = this.app3d.add_spheres;
                    bond = this.app3d.add_cylinders;
                    break;
                //case 3:
                //    radii = utils.mapper(syms, this.atom_cr)
                //                    .map(function(x) { return x * 0.5; });
                //    atom = this.app3d.add_points;
                //    bond = this.app3d.add_lines;
                //    break;
                //case 4:
                //    r = 0.6
                //    radii = r+0.05;
                //    atom = this.app3d.add_spheres;
                //    bond = this.app3d.add_cylinders;
                //    break;
            }
        } else {
            if (this.app3d.selected.length > 0) {
                this.clearSelected();
            };
            radii = utils.mapper(syms, this.atom_cr)
                            .map(function(x) { return x * 0.5; });
            atom = this.app3d.add_points;
            bond = this.app3d.add_lines;
        }
        var labels = utils.mapper(syms, this.atom_l);
        var entry = labels.entries();
        var a = [];
        for ( let e of entry ) { a.push(e[1]+e[0].toString()); }
        labels = a;
        this.app3d.meshes["atom"] = atom(
            this.atom_x[fdx], this.atom_y[fdx],
            this.atom_z[fdx], colrs, radii, labels);
        if (this.two_b0.length !== 0) {
            this.app3d.meshes["two"] = (bond != null) ? bond(
                this.two_b0[fdx], this.two_b1[fdx],
                this.atom_x[fdx], this.atom_y[fdx],
                this.atom_z[fdx], colrs, r) : null ;
        };
        this.app3d.add_meshes();
    },

    add_field: function() {
        this.app3d.clear_meshes("field");
        var fldx = this.model.get("field_idx");
        var fdx = this.model.get("frame_idx");
        var fps = this.field_p[fdx][fldx];
        if ((!this.model.get("field_show")) ||
            (fldx === "null") ||
            (typeof fps === "undefined")) { return }
        var idx = this.field_i[fdx][fldx];
        var that = this;
        if (typeof this.field_v[idx] === "string") {
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
        var fldx = this.model.get("field_idx");
        // Specifically test for string null
        var fdx = this.model.get("frame_idx");
        var idx = this.field_i[fdx][fldx];
        var fps = this.field_p[fdx][fldx];
        if ((!this.model.get("cont_show")) ||
            (fldx === "null") ||
            (typeof fps === "undefined")) { return }
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
        // Additionally adds the unit cell
        this.app3d.clear_meshes("generic");
        if (this.model.get("axis")) {
            this.app3d.meshes["generic"] = this.app3d.add_unit_axis(
                this.model.get("atom_3d"));
        };
        this.app3d.add_meshes("generic");
    },

    get_tensor: function(fdx, tdx) {
        var t = this.tensor_d[fdx][tdx];
        return [[t["xx"], t["xy"], t["xz"]],
                [t["yx"], t["yy"], t["yz"]],
                [t["zx"], t["zy"], t["zz"]]];
    },

    color_tensor: function() {
        var tdx = this.model.get("tidx");
        var fdx = this.model.get("frame_idx");
        var color;
        for ( var property in this.tensor_d[fdx] ) {
            if ( this.tensor_d[fdx].hasOwnProperty( property ) ) {
                if ( parseInt(property) === tdx ) { color = 0xafafaf; }
                else { color = 0x000000; }
                if ( this.model.get("tens") ) {
                    this.app3d.meshes["tensor"+property][0].children[0].material.color.setHex(color);
                }
            }
        }
    },

    add_tensor: function() {
        var scaling = this.model.get("scale");
        var fdx = this.model.get("frame_idx");
        for ( var property in this.tensor_d[fdx] ) {
            if ( this.tensor_d[fdx].hasOwnProperty( property ) ) {
                this.app3d.clear_meshes("tensor"+property);
                var adx = this.tensor_d[fdx][property]["atom"];
                if ( this.model.get("tens") ) {
                    this.app3d.meshes["tensor"+property] =
                                this.app3d.add_tensor_surface(
                                    this.get_tensor(fdx, property),
                                    this.colors(),
                                    this.atom_x[fdx][adx],
                                    this.atom_y[fdx][adx],
                                    this.atom_z[fdx][adx],
                                    scaling,
                                    this.tensor_d[fdx][property]["label"]+" tensor "+property.toString());
                }
                this.app3d.add_meshes("tensor"+property);
            }
        }
        //if (this.model.get("tens")) {this.color_tensor();}
    },

    events: {
        "click": "handleClick"
    },

    handleClick(event) {
        // Handles click event to write data to a Python traitlet value
        event.preventDefault();
        var idx = this.app3d.selected.map(function(obj) {return obj.name;});
        var type = this.app3d.selected.map(function(obj) {return obj.geometry.type;});
        this.model.set("selected", {idx, type}); //Types must match exactly
        this.touch();
    },

    clearSelected() {
        this.app3d.reset_colors();
        this.app3d.selected = [];
        this.model.set("selected", {});
        this.touch();
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
        this.listenTo(this.model, "change:tens", this.add_tensor);
        this.listenTo(this.model, "change:scale", this.add_tensor);
        this.listenTo(this.model, "change:tidx", this.color_tensor);
        this.listenTo(this.model, "change:fill_idx", this.add_atom);
        this.listenTo(this.model, "change:bond_r", this.add_atom);
        this.listenTo(this.model, "change:clear_selected", this.clearSelected);
    }

});


module.exports = {
    UniverseSceneModel: UniverseSceneModel,
    UniverseSceneView: UniverseSceneView
}
