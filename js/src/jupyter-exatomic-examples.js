// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=============
jupyter-exatomic-examples.js
=============
Example applications called when an empty container widget is rendered in a
Jupyter notebook environment.
*/

"use strict";
var THREE = require("three");
var base = require("./jupyter-exatomic-base.js");
var utils = require("./jupyter-exatomic-utils.js");


var TestSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "TestSceneModel",
            _view_name: "TestSceneView",
            geom: true,
            field: "null",
            field_ml: 0
        })
    }

});

var TestSceneView = base.ExatomicSceneView.extend({

    init: function() {
        TestSceneView.__super__.init.apply(this);
        this.init_listeners();
        this.add_geometry();
        this.animation();
    },

    add_geometry: function(color) {
        this.clear_meshes("generic");
        if (this.model.get("geom")) {
            this.meshes["generic"] = this.app3d.test_mesh();
        };
        this.add_meshes("generic");
    },

    add_field: function() {
        this.clear_meshes("field");
        var fps = this.get_fps();
        var ars = utils.gen_field_arrays(fps);
        var func = utils[this.model.get("field")];
        var iso = this.model.get("field_iso");
        var tf = utils.scalar_field(ars, func);
        this.meshes["field"] = this.app3d.add_scalar_field(tf, iso);
        this.add_meshes("field");
    },

    init_listeners: function() {
        TestSceneView.__super__.init_listeners.apply(this);
        this.listenTo(this.model, "change:geom", this.add_geometry);
        this.listenTo(this.model, "change:field", this.add_field);
    },

});


var TestUniverseSceneModel = base.ExatomicSceneModel.extend({

    defaults: _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
        _model_name: "TestUniverseSceneModel",
        _view_name: "TestUniverseSceneView",
        field: "Hydrogenic",
        field_iso: 0.005,
        field_kind: "1s",
        field_ox: -30.0,
        field_oy: -30.0,
        field_oz: -30.0,
        field_fx: 30.0,
        field_fy: 30.0,
        field_fz: 30.0,
        field_nx: 31,
        field_ny: 31,
        field_nz: 31
    })

});

var TestUniverseSceneView = base.ExatomicSceneView.extend({

    init: function() {
        TestUniverseSceneView.__super__.init.apply(this);
        this.app3d.set_camera({"x": 40.0, "y": 40.0, "z": 40.0});
        this.init_listeners();
        this.add_field();
        this.animation();
    },

    add_field: function() {
        this.clear_meshes("field");
        var field = this.model.get("field");
        var kind = this.model.get("field_kind");
        var iso = this.model.get("field_iso");
        var fps = this.get_fps();
        var ars = utils.gen_field_arrays(fps);
        if (field === 'SolidHarmonic') {
            var tf = utils[field](ars, kind, this.model.get("field_ml"));
        } else {
            var tf = utils[field](ars, kind);
        };
        this.meshes["field"] = this.app3d.add_scalar_field(tf, iso, 2);
        this.add_meshes("field");
    },

    init_listeners: function() {
        TestUniverseSceneView.__super__.init_listeners.apply(this);
        this.listenTo(this.model, "change:field_kind", this.add_field);
        this.listenTo(this.model, "change:field_ml", this.add_field);
    },

});


var TestContainerModel = base.ExatomicBoxModel.extend({
    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "TestContainerModel",
        _view_name: "TestContainerView"
    })
});

var TestContainerView = base.ExatomicBoxView.extend({});


var TestUniverseModel = base.ExatomicBoxModel.extend({
    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "TestUniverseModel",
        _view_name: "TestUniverseView"
    })
});

var TestUniverseView = base.ExatomicBoxView.extend({});


module.exports = {
    TestSceneModel: TestSceneModel,
    TestSceneView: TestSceneView,
    TestContainerModel: TestContainerModel,
    TestContainerView: TestContainerView,
    TestUniverseSceneModel: TestUniverseSceneModel,
    TestUniverseSceneView: TestUniverseSceneView,
    TestUniverseModel: TestUniverseModel,
    TestUniverseView: TestUniverseView
}
