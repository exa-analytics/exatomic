// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
scene.ts
=================
A 3D scene for exatomic

*/

import * as widgets from '@jupyter-widgets/base'
import * as pkg from '../package.json'

// const version: string = require('../package.json').version
const semver: string = `^${pkg.version}`

import * as three from 'three'
import * as TrackBallControls from 'three-trackballcontrols'

export class SceneModel extends widgets.DOMWidgetModel {
    defaults() {
        return {
            ...super.defaults(),
            _model_name: 'SceneModel',
            _view_name: 'SceneView',
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: 'exatomic',
            _view_module: 'exatomic',
            width: 200,
            height: 200,
        }
    }
}


export class SceneView extends widgets.DOMWidgetView {
    renderer: three.WebGLRenderer
    camera: three.PerspectiveCamera
    controls: any // TrackBallControls


    initialize(parameters: any) {
        super.initialize(parameters)
        console.log(this.model.get('_model_module_version'))
        this.renderer = new three.WebGLRenderer({
            antialias: true,
            alpha: true
        })
        this.renderer.autoClear = false
        this.renderer.shadowMap.enabled = true
        this.renderer.shadowMap.type = three.PCFSoftShadowMap
        this.el.appendChild(this.renderer.domElement)

        // console.log(this.model.get('width'))
        // console.log(this.model.get('height'))

        this.camera = new three.PerspectiveCamera(
            35, this.model.get('width') / this.model.get('height'), 1, 100000)


        this.controls = new TrackBallControls(
            this.camera,
            this.renderer.domElement
        )
        this.controls.rotateSpeed = 10.0
        this.controls.zoomSpeed = 5.0
        this.controls.panSpeed = 0.5
        this.controls.noZoom = false
        this.controls.noPan = false
        this.controls.staticMoving = true
        this.controls.dynamicDampingFactor = 0.3
        this.controls.keys = [65, 83, 68]
        this.controls.target = new three.Vector3(0.0, 0.0, 0.0)

        console.log(this)
        this.initListeners()
        this.resize()
    }

    initListeners() {
        this.listenTo(this.model, 'change:flag', this.updateFlag)
        this.listenTo(this.model, 'change:width', this.resize)
        this.listenTo(this.model, 'change:height', this.resize)
    }

    updateFlag() {
        this.resize()
    }

    resize() {
        let w: number = this.model.get('width')
        let h: number = this.model.get('height')
        console.log("resize (w, h, el.w, el.h)", w, h)
        this.renderer.setSize(w, h)
        this.camera.aspect = w / h
        this.camera.updateProjectionMatrix()
        this.camera.updateMatrix()
        this.controls.handleResize()
    }
}
