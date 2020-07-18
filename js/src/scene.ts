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
	scene: three.Scene
    camera: three.PerspectiveCamera
    renderer: three.WebGLRenderer
    controls: any // TrackBallControls
	hudscene: three.Scene
	hudcamera: three.OrthographicCamera
	raycaster: three.Raycaster
	mouse: three.Vector2
    hudcanvas: any // "canvas"
	promises: Promise<any>

    initialize(parameters: any) {
        super.initialize(parameters)
        this.initListeners()
        this.promises = this.init()
        // this.resize()
    }

    init() {
        return Promise.all([
            this.initScene(),
            this.initRenderer(),
            this.initHud(),
        ]).then(() => {
            this.resize()
            this.initObj()
		})
    }

	initScene() {
        let scene = new three.Scene()
        let amlight = new three.AmbientLight(0xdddddd, 0.5)
        let dlight0 = new three.DirectionalLight(0xdddddd, 0.3)
        dlight0.position.set(-1000, -1000, -1000)
        let sunlight = new three.SpotLight(0xdddddd, 0.3, 0, Math.PI/2)
        sunlight.position.set(1000, 1000, 1000)
        sunlight.castShadow = true
        // sunlight.shadow = new three.LightShadow(
        //     new three.PerspectiveCamera(30, 1, 1500, 5000)
        // )
        sunlight.shadow.bias = 0.
        scene.add(amlight)
        scene.add(dlight0)
        scene.add(sunlight)
        return Promise.resolve(scene).then((scene) => {
            this.scene = scene
        })
	}

	initRenderer() {
		return Promise.all([
			Promise.resolve(new three.WebGLRenderer({
            	antialias: true,
            	alpha: true
        	})).then((renderer) => {
				renderer.autoClear = false
				renderer.shadowMap.enabled = true
				renderer.shadowMap.type = three.PCFSoftShadowMap
				this.renderer = renderer
				this.el.appendChild(this.renderer.domElement)
			}),
			Promise.resolve(new three.PerspectiveCamera(
            	35, this.model.get('width') / this.model.get('height'), 1, 100000
			)).then((camera) => {
				this.camera = camera
			})
		]).then(() => {
			Promise.resolve(new TrackBallControls(
            	this.camera, this.renderer.domElement
			)).then((controls) => {
                controls.rotateSpeed = 10.0
                controls.zoomSpeed = 5.0
                controls.panSpeed = 0.5
                controls.noZoom = false
                controls.noPan = false
                controls.staticMoving = true
                controls.dynamicDampingFactor = 0.3
                controls.keys = [65, 83, 68]
                controls.target = new three.Vector3(0.0, 0.0, 0.0)
				this.controls = controls
				this.controls.addEventListener("change", this.render.bind(this))
			})
		})
	}


    initHud() {
        return Promise.all([
            Promise.resolve(new three.Raycaster()).then((raycaster) => {
                this.raycaster = raycaster
            }),
            Promise.resolve(new three.Scene()).then((hudscene) => {
                this.hudscene = hudscene
            }),
            Promise.resolve(new three.OrthographicCamera(
                -this.model.get('width') / 2,
				 this.model.get('width') / 2,
				 this.model.get('height') / 2,
				-this.model.get('height') / 2,
				1, 1500
            )).then((cam) => {
                cam.position.z = 1000
                this.hudcamera = cam
            }),
            Promise.resolve(new three.Vector2()).then((mouse) => {
                this.mouse = mouse
            }),
            Promise.resolve(
                <HTMLCanvasElement> document.createElement("canvas")
            ).then((canvas) => {
                canvas.width = 1024
                canvas.height = 1024
                this.hudcanvas = canvas
            })
        ])
    }

	initObj() {
		let geom = new three.IcosahedronGeometry(2, 1)
        let mat0 = new three.MeshBasicMaterial({
			color: 0x000000,
			wireframe: true,
		})
		let mesh0 = new three.Mesh(geom, mat0)
		mesh0.position.set(0, 0, -3)
		mesh0.name = "Icosahedron 0"

        let mat1 = new three.MeshBasicMaterial({
			color: 0xFF0000,
			wireframe: true,
		})
		let mesh1 = new three.Mesh(geom, mat1)
		mesh1.position.set(0, 0, 3)
		mesh1.name = "Icosahedron 1"
		this.scene.add(mesh0)
		this.scene.add(mesh1)
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

    render() {
		return this.promises.then(this.animate.bind(this))
            //.then(this.addAtom.bind(this))
            //.then(this.setCameraFromScene.bind(this))
    }

    paint() {
        this.renderer.clear()
        this.renderer.render(this.scene, this.camera)
        this.renderer.clearDepth()
        this.renderer.render(this.hudscene, this.hudcamera)
    }

    animate() {
        let id = requestAnimationFrame(this.animate.bind(this))
        if (this.renderer !== null) {
            // this.resize()
            this.controls.update()
            this.paint()
        } else {
            console.log("Cancelling animation frame.")
            window.cancelAnimationFrame(id)
        }
    }

}
