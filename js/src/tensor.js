// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
=============
tensor.js
=============
Example applications called when an empty container widget is rendered in a
Jupyter notebook environment.
*/

"use strict";
var base = require("./base");
var utils = require("./utils");
var _ = require('underscore');
// var App3D = require("./appthree.js").App3D;


export TensorSceneModel extends base.ExatomicSceneModel {

    defaults() {
        return {
            ...super.defaults(),
            _model_name: "TensorSceneModel",
            _view_name: "TensorSceneView",
            geom: true,
            field: "null",
            field_ml: 0,
            tensor: [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        }
    }

}



export TensorSceneView extends base.ExatomicSceneView {

    init() {
        super.init()
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.generate_tensor.bind(this))
            .then(this.app3d.set_camera_from_scene.bind(this.app3d))
    }

    generate_tensor() {
        this.app3d.clear_meshes("generic")
        if (this.model.get("geom")) {
            this.app3d.meshes["generic"] =
                   this.app3d.add_tensor_surface(this.get_tensor(), this.colors())
        }
        this.app3d.add_meshes("generic")
    }

    get_tensor() {
        return [
            [this.model.get("txx"), this.model.get("txy"), this.model.get("txz")],
            [this.model.get("tyx"), this.model.get("tyy"), this.model.get("tyz")],
            [this.model.get("tzx"), this.model.get("tzy"), this.model.get("tzz")],
        ]
    }

    init_listeners() {
        super.init_listeners()
        this.listenTo(this.model, "change:geom", this.generate_tensor)
        this.listenTo(this.model, "change:txx", this.generate_tensor)
        this.listenTo(this.model, "change:txy", this.generate_tensor)
        this.listenTo(this.model, "change:txz", this.generate_tensor)
        this.listenTo(this.model, "change:tyx", this.generate_tensor)
        this.listenTo(this.model, "change:tyy", this.generate_tensor)
        this.listenTo(this.model, "change:tyz", this.generate_tensor)
        this.listenTo(this.model, "change:tzx", this.generate_tensor)
        this.listenTo(this.model, "change:tzy", this.generate_tensor)
        this.listenTo(this.model, "change:tzz", this.generate_tensor)
        this.listenTo(this.modle, "change:tdx", this.generate_tensor)
    }

}


class TensorContainerModel extends base.ExatomicBoxModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: "TensorContainerModel",
            _view_name: "TensorContainerView"
    }
}

class TensorContainerView extends base.ExatomicBoxView {}
