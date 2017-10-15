
"use strict";
var _ = require("underscore");
var base = require("./jupyter-exatomic-base.js");
var utils = require("./jupyter-exatomic-utils.js");

// Taken from another file -- check imports and package references
//var ExatomicWidgetModel = widgets.WidgetModel.extend({
//
//    defaults: function() {
//        return _.extend({}, widgets.WidgetModel.prototype.defaults, {
//            _model_module: "jupyter-exatomic",
//            _view_module: "jupyter-exatomic",
//            _model_name: "ExatomicWidgetModel",
//            _view_name: "ExatomicWidgetView"
//        })
//    }
//
//});
//
//var ExatomicWidgetView = widgets.WidgetView.extend({
//
//});



var SmallverseSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "SmallverseSceneModel",
            _view_name: "SmallverseSceneView",
            frame_idx: 0,
            field: null,
            frame: null,
            atom: null,
            two: null
        })
    }

},  {
    serializers: _.extend({}, {
        atom: {deserialize: base.unpack_models},
        field: {deserialize: base.unpack_models},
        frame: {deserialize: base.unpack_models},
        two: {deserialize: base.unpack_models}
        }, base.ExatomicSceneModel.serializers)
});

var SmallverseSceneView = base.ExatomicSceneView.extend({

    init: function() {
        SmallverseSceneView.__super__.init.apply(this);
        this.app3d.set_camera({"x": 40.0, "y": 40.0, "z": 40.0});
        console.log(this.model);
        this.init_listeners();
        this.add_atom();
        this.animation();
    },

    add_atom: function() {
        // Smallverse
        this.clear_meshes("atom");
        this.clear_meshes("two");
        var s = this.model.get("atom").attributes.s;
        var c = utils.mapper(s, this.model.get("atom").attributes.c);
        var r = utils.mapper(s, this.model.get("atom").attributes.r);
        if (this.model.get("atom_spheres")) {
            var atom = this.app3d.add_spheres;
            var bond = this.app3d.add_cylinders;
        } else {
            var atom = this.app3d.add_points;
            var bond = this.app3d.add_lines;
        };
        this.meshes["atom"] = atom(this.model.get("atom").attributes.x,
                                   this.model.get("atom").attributes.y,
                                   this.model.get("atom").attributes.z,
                                   c, r);
        if (this.model.get("two") !== null) {
            this.meshes["two"] = bond(this.model.get("two").attributes.b0,
                                      this.model.get("two").attributes.b1,
                                      this.model.get("atom").attributes.x,
                                      this.model.get("atom").attributes.y,
                                      this.model.get("atom").attributes.z,
                                      c);
        };
        this.add_meshes();
    },

    add_field: function() {
        // Smallverse
        this.clear_meshes("field");
        console.log(this.model.get("field_iso"));
        console.log(this.model.get("field"));
        var iso = this.model.get("field_iso");
        var ars = utils.gen_field_arrays(this.model.get("field").attributes.params);
        var tf = utils.scalar_field(ars, this.model.get("field").attributes.values);
        this.meshes["field"] = this.app3d.add_scalar_field(tf, iso, 2);
        this.add_meshes("field");
    },

        // Smallverse
    do_nothing: function() {console.log("I'm doing nothing.")},

    init_listeners: function() {
        SmallverseSceneView.__super__.init_listeners.apply(this);
        this.listenTo(this.model, "change:frame_idx", this.add_atom);
        this.listenTo(this.model, "change:atom_spheres", this.add_atom);
        this.listenTo(this.model, "change:field_idx", this.add_field);
    }

});





var SmallverseWidgetModel = base.ExatomicBoxModel.extend({
    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "SmallverseWidgetModel",
        _view_name: "SmallverseWidgetView",
    })
});

var SmallverseWidgetView = base.ExatomicBoxView.extend({});


var AtomWidgetModel = base.ExatomicWidgetModel.extend({
    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
        _model_name: "AtomWidgetModel",
        _view_name: "AtomWidgetView",
    })
});

var AtomWidgetView = base.ExatomicWidgetView.extend({});


var FrameWidgetModel = base.ExatomicWidgetModel.extend({
    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
        _model_name: "FrameWidgetModel",
        _view_name: "FrameWidgetView"
    })
});

var FrameWidgetView = base.ExatomicWidgetView.extend({});



var FieldWidgetModel = base.ExatomicWidgetModel.extend({
    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
        _model_name: "FieldWidgetModel",
        _view_name: "FieldWidgetView"
    })
});

var FieldWidgetView = base.ExatomicWidgetView.extend({});


var TwoWidgetModel = base.ExatomicWidgetModel.extend({
    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
        _model_name: "TwoWidgetModel",
        _view_name: "TwoWidgetView"
    })
});

var TwoWidgetView = base.ExatomicWidgetView.extend({});

//var AllAtomWidgetModel = base.ExatomicWidgetModel.extend({
//    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
//        _model_name: "AllAtomWidgetModel",
//        _view_name: "AllAtomWidgetView"
//    })
//});
//
//var AllAtomWidgetView = base.ExatomicWidgetView.extend({});
//
//
//var AllFieldWidgetModel = base.ExatomicWidgetModel.extend({
//    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
//        _model_name: "AllFieldWidgetModel",
//        _view_name: "AllFieldWidgetView"
//    })
//});
//
//var AllFieldWidgetView = base.ExatomicWidgetView.extend({});
//
//
//var AllTwoWidgetModel = base.ExatomicWidgetModel.extend({
//    defaults: _.extend({}, base.ExatomicWidgetModel.prototype.defaults, {
//        _model_name: "AllTwoWidgetModel",
//        _view_name: "AllTwoWidgetView"
//    })
//});
//
//var AllTwoWidgetView = base.ExatomicWidgetView.extend({});

module.exports = {
    SmallverseWidgetModel: SmallverseWidgetModel,
    SmallverseWidgetView: SmallverseWidgetView,
    SmallverseSceneModel: SmallverseSceneModel,
    SmallverseSceneView: SmallverseSceneView,
    AtomWidgetModel: AtomWidgetModel,
    AtomWidgetView: AtomWidgetView,
    FieldWidgetModel: FieldWidgetModel,
    FieldWidgetView: FieldWidgetView,
    TwoWidgetModel: TwoWidgetModel,
    TwoWidgetView: TwoWidgetView
    // All the All widgets...
    //ExatomicWidgetModel: ExatomicWidgetModel,
    //ExatomicWidgetView: ExatomicWidgetView,
}
