// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=============
tensor.js
=============
Example applications called when an empty container widget is rendered in a
Jupyter notebook environment.
*/

const base = require('./base')

export class TensorSceneModel extends base.ExatomicSceneModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'TensorSceneModel',
            _view_name: 'TensorSceneView',
            geom: true,
            field: 'null',
            field_ml: 0,
            tensor: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

export class TensorSceneView extends base.ExatomicSceneView {
    init() {
        super.init()
        this.three_promises = this.app3d.finalize(this.three_promises)
            .then(this.generateTensor.bind(this))
            .then(this.app3d.set_camera_from_scene.bind(this.app3d))
    }

    generateTensor() {
        this.app3d.clear_meshes('generic')
        if (this.model.get('geom')) {
            this.app3d.meshes.generic = this.app3d.add_tensor_surface(
                this.getTensor(),
                this.colors(),
            )
        }
        this.app3d.add_meshes('generic')
    }

    getTensor() {
        return [
            [this.model.get('txx'), this.model.get('txy'), this.model.get('txz')],
            [this.model.get('tyx'), this.model.get('tyy'), this.model.get('tyz')],
            [this.model.get('tzx'), this.model.get('tzy'), this.model.get('tzz')],
        ]
    }

    initListeners() {
        super.initListeners()
        this.listenTo(this.model, 'change:geom', this.generateTensor)
        this.listenTo(this.model, 'change:txx', this.generateTensor)
        this.listenTo(this.model, 'change:txy', this.generateTensor)
        this.listenTo(this.model, 'change:txz', this.generateTensor)
        this.listenTo(this.model, 'change:tyx', this.generateTensor)
        this.listenTo(this.model, 'change:tyy', this.generateTensor)
        this.listenTo(this.model, 'change:tyz', this.generateTensor)
        this.listenTo(this.model, 'change:tzx', this.generateTensor)
        this.listenTo(this.model, 'change:tzy', this.generateTensor)
        this.listenTo(this.model, 'change:tzz', this.generateTensor)
        this.listenTo(this.modle, 'change:tdx', this.generateTensor)
    }
}

export class TensorContainerModel extends base.ExatomicBoxModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'TensorContainerModel',
            _view_name: 'TensorContainerView',
        }
    }
}

export class TensorContainerView extends base.ExatomicBoxView {
}
