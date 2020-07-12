// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
base.js
=================
JavaScript "frontend" complement of exatomic"s Universe
for use within the Jupyter notebook interface.
*/

const widgets = require('@jupyter-widgets/base')
const control = require('@jupyter-widgets/controls')
const three = require('./appthree')
const utils = require('./utils')
const ver = require('../package.json').version

const semver = `^${ver}`
// eslint-disable-next-line no-console
console.log(`exatomic JS version: ${semver}`)

export class ExatomicBoxModel extends control.BoxModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'ExatomicBoxModel',
            _view_name: 'ExatomicBoxView',
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: 'exatomic',
            _view_module: 'exatomic',
            linked: false,
        }
    }
}

export class ExatomicBoxView extends control.BoxView {
    initialize(parameters) {
        super.initialize(parameters)
        this.init()
    }

    init() {
        this.initListeners()
        const that = this
        this.displayed.then(() => {
            that.scene_ps = that.children_views.views[1].then((vbox) => {
                const hboxs = vbox.children_views.views
                const promises = Promise.all(hboxs).then((hbox) => {
                    const subpromises = []
                    for (let i = 0; i < hbox.length; i += 1) {
                        const scns = hbox[i].children_views.views
                        for (let j = 0; j < scns.length; j += 1) {
                            subpromises.push(scns[j])
                        }
                    }
                    return Promise.all(subpromises).then((p) => p)
                })
                return promises
            })
            that.scene_ps.then((p) => {
                for (let i = 0; i < p.length; i += 1) {
                    p[i].resize()
                }
            })
        })
    }

    linkControls() {
        // TODO :: Instead of referencing the first camera object
        //      :: just set camera.rotation (and camera.zoom??) to
        //      :: copy original camera.
        //      :: e.g. -- camera[i].rotation.copy(camera[0])
        let i; let
            app
        const that = this
        this.scene_ps.then((views) => {
            if (that.model.get('linked')) {
                const idxs = that.model.get('active_scene_indices')
                // const { controls } = views[idxs[0]].app3d
                const { camera } = views[idxs[0]].app3d
                for (i = 1; i < idxs.length; i += 1) {
                    app = views[idxs[i]].app3d
                    app.camera = camera
                    app.controls = app.init_controls()
                    app.controls.addEventListener('change', app.render.bind(app))
                }
            } else {
                for (i = 0; i < views.length; i += 1) {
                    app = views[i].app3d
                    app.camera = app.camera.clone()
                    app.controls = app.init_controls()
                    app.controls.addEventListener('change', app.render.bind(app))
                }
            }
        })
    }

    initListeners() {
        this.listenTo(this.model, 'change:linked', this.linkControls)
    }
}

export class ExatomicSceneModel extends widgets.DOMWidgetModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'ExatomicSceneModel',
            _view_name: 'ExatomicSceneView',
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: 'exatomic',
            _view_module: 'exatomic',
        }
    }
}

export class ExatomicSceneView extends widgets.DOMWidgetView {
    initialize(parameters) {
        super.initialize(parameters)
        this.initListeners()
        this.init()
    }

    init() {
        let func
        window.addEventListener('resize', this.resize.bind(this))
        this.app3d = new three.App3D(this)
        this.three_promises = this.app3d.init_promise()
        if (this.model.get('uni')) {
            func = this.addField
        } else {
            func = this.addGeometry
        }
        this.three_promises.then(func.bind(this))
            .then(this.app3d.set_camera.bind(this.app3d))
    }

    resize() {
        // sometimes during window resize these are 0
        const w = this.el.offsetWidth || 200
        const h = this.el.offsetHeight || 200
        this.model.set('w', w)
        // threejs canvas is 5 smaller than div
        this.model.set('h', h - 5)
    }

    render() {
        return this.app3d.finalize(this.three_promises)
    }

    addGeometry() {
        this.app3d.clear_meshes('generic')
        if (this.model.get('geom')) {
            this.app3d.meshes.generic = this.app3d.test_mesh()
            this.app3d.add_meshes('generic')
        }
    }

    colors() {
        return {
            pos: this.model.get('field_pos'),
            neg: this.model.get('field_neg'),
        }
    }

    addField() {
        this.app3d.clear_meshes('field')
        if (this.model.get('uni')) {
            let name; let tf
            const field = this.model.get('field')
            const kind = this.model.get('field_kind')
            const ars = utils.gen_field_arrays(this.getFps())
            const func = utils[field]
            if (field === 'SolidHarmonic') {
                const fml = this.model.get('field_ml')
                tf = func(ars, kind, fml)
                name = `Sol.Har.,${kind},${fml}`
            } else {
                tf = func(ars, kind)
                name = `${field},${kind}`
            }
            this.app3d.meshes.field = this.app3d.add_scalar_field(
                tf, this.model.get('field_iso'),
                this.model.get('field_o'), 2,
                this.colors(),
            )
            for (let i = 0; i < this.app3d.meshes.field.length; i += 1) {
                this.app3d.meshes.field[i].name = name
            }
        } else {
            this.app3d.meshes.field = this.app3d.add_scalar_field(
                utils.scalar_field(
                    utils.gen_field_arrays(this.getFps()),
                    utils[this.model.get('field')],
                ),
                this.model.get('field_iso'),
                this.model.get('field_o'),
            )
            this.app3d.meshes.field[0].name = this.model.get('field')
        }
        this.app3d.add_meshes('field')
    }

    updateField() {
        const meshes = this.app3d.meshes.field
        for (let i = 0; i < meshes.length; i += 1) {
            meshes[i].material.transparent = true
            meshes[i].material.opacity = this.model.get('field_o')
            meshes[i].material.needsUpdate = true
        }
    }

    getFps() {
        const fps = {
            ox: this.model.get('field_ox'),
            oy: this.model.get('field_oy'),
            oz: this.model.get('field_oz'),
            nx: this.model.get('field_nx'),
            ny: this.model.get('field_ny'),
            nz: this.model.get('field_nz'),
            fx: this.model.get('field_fx'),
            fy: this.model.get('field_fy'),
            fz: this.model.get('field_fz'),
        }
        fps.dx = (fps.fx - fps.ox) / (fps.nx - 1)
        fps.dy = (fps.fy - fps.oy) / (fps.ny - 1)
        fps.dz = (fps.fz - fps.oz) / (fps.nz - 1)
        return fps
    }

    clearMeshes() {
        this.app3d.clear_meshes()
    }

    save() {
        this.send({ type: 'image', content: this.app3d.save() })
    }

    saveCamera() {
        this.send({ type: 'camera', content: this.app3d.camera.toJSON() })
    }

    handleCustomMsg(msg) {
        if (msg.type === 'close') { this.app3d.close() }
        if (msg.type === 'camera') {
            this.app3d.set_camera_from_camera(msg.content)
        }
    }

    initListeners() {
        // The basics
        this.listenTo(this.model, 'change:clear', this.clearMeshes)
        this.listenTo(this.model, 'change:save', this.save)
        this.listenTo(this.model, 'change:save_cam', this.saveCamera)
        this.listenTo(this.model, 'msg:custom', this.handleCustomMsg)
        this.listenTo(this.model, 'change:geom', this.addGeometry)
        // Field stuff
        if (!this.model.get('uni')) {
            this.listenTo(this.model, 'change:field', this.addField)
        }
        this.listenTo(this.model, 'change:field_kind', this.addField)
        this.listenTo(this.model, 'change:field_ml', this.addField)
        this.listenTo(this.model, 'change:field_o', this.updateField)
        this.listenTo(this.model, 'change:field_nx', this.addField)
        this.listenTo(this.model, 'change:field_ny', this.addField)
        this.listenTo(this.model, 'change:field_nz', this.addField)
        this.listenTo(this.model, 'change:field_iso', this.addField)
    }
}
