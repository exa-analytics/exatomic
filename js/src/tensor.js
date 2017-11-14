// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=============
tensor.js
=============
Example applications called when an empty container widget is rendered in a
Jupyter notebook environment.
*/

"use strict";
var base = require("./base.js");
var utils = require("./utils.js");
// var App3D = require("./appthree.js").App3D;


var TensorSceneModel = base.ExatomicSceneModel.extend({

    defaults: function() {
        return _.extend({}, base.ExatomicSceneModel.prototype.defaults, {
            _model_name: "TensorSceneModel",
            _view_name: "TensorSceneView",
            geom: true,
            field: "null",
            field_ml: 0,
            tensor: [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        })
    }

});



var TensorSceneView = base.ExatomicSceneView.extend({

    init: function() {
        base.ExatomicSceneView.prototype.init.apply(this);
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.add_surface.bind(this))
            .then(this.app3d.set_camera_from_scene.bind(this.app3d));
    },

    add_surface: function() {
        this.app3d.clear_meshes("generic");
        if (this.model.get("geom")) {
            this.app3d.meshes["generic"] =
                this.app3d.add_tensor_surface( this.model.get("tensor") );
        };
        this.app3d.add_meshes("generic");
    },

    generate_tensor: function() {
        var tensor = this.model.get("tensor");
        console.log(this.model.get("generate"));
        console.log(tensor[0]);
        this.app3d.clear_meshes("generic");
        this.app3d.meshes["generic"] =
               this.app3d.add_tensor_surface( tensor );
        this.app3d.add_meshes("generic");
    },

    init_listeners: function() {
        base.ExatomicSceneView.prototype.init_listeners.call(this);
        this.listenTo(this.model, "change:geom", this.add_surface);
        this.listenTo(this.model, "change:generate", this.generate_tensor);
    },

});


var TensorContainerModel = base.ExatomicBoxModel.extend({
    defaults: _.extend({}, base.ExatomicBoxModel.prototype.defaults, {
        _model_name: "TensorContainerModel",
        _view_name: "TensorContainerView"
    })
});

var TensorContainerView = base.ExatomicBoxView.extend({});


module.exports = {
    TensorSceneModel: TensorSceneModel,
    TensorSceneView: TensorSceneView,
    TensorContainerModel: TensorContainerModel,
    TensorContainerView: TensorContainerView
}
