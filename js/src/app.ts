// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
app.js
=================
A three.js adapter to exatomic

*/

import * as three from 'three'
import * as TrackBallControls from 'three-trackballcontrols'

export class App {
    view: any
    meshes: Object // [Array<three.Mesh>]
    selected: Array<three.Mesh>
    renderer: three.WebGLRenderer
    camera: three.PerspectiveCamera
    controls: any  // TrackBallControls
    w: number
    h: number

    constructor(view: any) {
        this.view = view
        this.w = 200
        this.h = 200
        this.meshes = {
            contour: [],
            frame: [],
            field: [],
            atom: [],
            test: [],
            two: [],
        }

        this.renderer = new three.WebGLRenderer({
            antialias: true,
            alpha: true
        })
        this.renderer.autoClear = false
        this.renderer.shadowMap.enabled = true
        this.renderer.shadowMap.type = three.PCFSoftShadowMap

        this.camera = new three.PerspectiveCamera(35, this.w / this.h, 1, 100000)


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

    }

}

