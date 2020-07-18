// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
scene.ts
=================
A 3D scene for exatomic

*/

import * as three from 'three'
import * as TrackBallControls from 'three-trackballcontrols'

import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base'
import { version } from '../package.json'

const semver = `^${version}`

export class SceneModel extends DOMWidgetModel {
    defaults(): any {
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

export class SceneView extends DOMWidgetView {
    scene: three.Scene

    camera: three.PerspectiveCamera

    renderer: three.WebGLRenderer | null

    controls: any // TrackBallControls

    hudscene: three.Scene

    hudcamera: three.OrthographicCamera

    raycaster: three.Raycaster

    mouse: three.Vector2

    hudcanvas: HTMLCanvasElement

    promises: Promise<any>

    initialize(parameters: any): void {
        super.initialize(parameters)
        this.initListeners()
        this.promises = this.init()
        this.displayed.then(() => {
            this.resize()
            this.setCameraFromScene()
        })
    }

    init(): Promise<any> {
        window.addEventListener('resize', this.resize.bind(this))
        return Promise.all([
            this.initScene(),
            this.initRenderer(),
            this.initHud(),
        ]).then(() => {
            this.initObj()
            this.finalizeMouseover()
            // this.resize()
            // this.setCameraFromScene()
        })
    }

    initScene() {
        const scene = new three.Scene()
        const amlight = new three.AmbientLight(0xdddddd, 0.5)
        const dlight0 = new three.DirectionalLight(0xdddddd, 0.3)
        const sunlight = new three.SpotLight(0xdddddd, 0.3, 0, Math.PI / 2)
        const shadowcam = new three.PerspectiveCamera(30, 1, 1500, 5000)
        dlight0.position.set(-1000, -1000, -1000)
        sunlight.position.set(1000, 1000, 1000)
        sunlight.castShadow = true
        sunlight.shadow = new three.SpotLightShadow(shadowcam)
        //    camera: shadowcam,
        //    isSpotLightShadow: true,
        //})
        sunlight.shadow.bias = 0.0
        scene.add(amlight)
        scene.add(dlight0)
        scene.add(sunlight)
        return Promise.resolve(scene).then((scn) => {
            this.scene = scn
        })
    }

    initRenderer(): any {
        return Promise.all([
            Promise.resolve(new three.WebGLRenderer({
                antialias: true,
                alpha: true,
            })).then((renderer) => {
                this.renderer = renderer
                this.renderer.autoClear = false
                this.renderer.shadowMap.enabled = true
                this.renderer.shadowMap.type = three.PCFSoftShadowMap
                this.el.appendChild(this.renderer.domElement)
            }),
            Promise.resolve(new three.PerspectiveCamera(
                35, this.model.get('width') / this.model.get('height'), 1, 100000,
            )).then((camera) => {
                this.camera = camera
            }),
        ]).then(() => {
            if (this.renderer !== null) {
                Promise.resolve(new TrackBallControls(
                    this.camera, this.renderer.domElement,
                )).then((controls) => {
                    this.controls = controls
                    this.controls.rotateSpeed = 10.0
                    this.controls.zoomSpeed = 5.0
                    this.controls.panSpeed = 0.5
                    this.controls.noZoom = false
                    this.controls.noPan = false
                    this.controls.staticMoving = true
                    this.controls.dynamicDampingFactor = 0.3
                    this.controls.keys = [65, 83, 68]
                    this.controls.target = new three.Vector3(0.0, 0.0, 0.0)
                    this.controls.addEventListener('change', this.render.bind(this))
                })
            }
        })
    }

    initHud(): any {
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
                1, 1500,
            )).then((camera) => {
                this.hudcamera = camera
                this.hudcamera.position.z = 1000
            }),
            Promise.resolve(new three.Vector2()).then((mouse) => {
                this.mouse = mouse
            }),
            Promise.resolve(
                <HTMLCanvasElement> document.createElement('canvas'),
            ).then((canvas) => {
                this.hudcanvas = canvas
                this.hudcanvas.width = 1024
                this.hudcanvas.height = 1024
            }),
        ])
    }

    initObj(): void {
        const geom = new three.IcosahedronGeometry(2, 1)
        const mat0 = new three.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true,
        })
        const mesh0 = new three.Mesh(geom, mat0)
        mesh0.position.set(0, 0, -3)
        mesh0.name = 'Icosahedron 0'

        const mat1 = new three.MeshBasicMaterial({
            color: 0xFF0000,
            wireframe: true,
        })
        const mesh1 = new three.Mesh(geom, mat1)
        mesh1.position.set(0, 0, 3)
        mesh1.name = 'Icosahedron 1'
        this.scene.add(mesh0)
        this.scene.add(mesh1)
    }

    initListeners(): void {
        this.listenTo(this.model, 'change:flag', this.updateFlag)
        this.listenTo(this.model, 'change:layout.width', this.resize)
        this.listenTo(this.model, 'change:layout.height', this.resize)
    }

    updateFlag(): void {
        this.resize()
    }

    resize(): void {
        const w: number = this.el.offsetWidth || this.model.get('width')
        let h: number = this.el.offsetHeight || this.model.get('height')
        // hack canvas in el is 6 smaller than el
        if (h === this.el.offsetHeight) { h -= 6 }
        console.log('resize (w, h, el.w, el.h)', w, h, this.el.offsetWidth, this.el.offsetHeight)
        console.log(this.el, this.model.get('layout'))
        console.log(this.el.getBoundingClientRect())
        console.log(this.el.canvas)
        if (this.renderer !== null) {
            this.renderer.setSize(w, h)
            this.camera.aspect = w / h
            this.camera.updateProjectionMatrix()
            this.camera.updateMatrix()
            this.hudcamera.left   = -w / 2
            this.hudcamera.right  =  w / 2
            this.hudcamera.top    =  h / 2
            this.hudcamera.bottom = -h / 2
            this.hudcamera.updateProjectionMatrix()
            this.controls.handleResize()
        }
    }

    render(): any {
        return this.promises.then(this.animate.bind(this))
    }

    paint(): void {
        if (this.renderer !== null) {
            this.controls.update()
            this.renderer.clear()
            this.renderer.render(this.scene, this.camera)
            this.renderer.clearDepth()
            this.renderer.render(this.hudscene, this.hudcamera)
        }
    }

    animate(): void {
        if (this.renderer !== null) {
            this.renderer.setAnimationLoop(this.paint.bind(this))
        }
    }

    setCameraFromScene(): void {
        const bbox = new three.Box3().setFromObject(this.scene)
        const { min } = bbox
        const { max } = bbox
        const ox = (max.x + min.x ) / 2
        const oy = (max.y + min.y ) / 2
        const oz = (max.z + min.z ) / 2
        const px = Math.max(2 * max.x, 30)
        const py = Math.max(2 * max.y, 30)
        const pz = Math.max(2 * max.z, 30)
        this.camera.position.set(px, py, pz)
        this.controls.target.setX(ox)
        this.controls.target.setY(oy)
        this.controls.target.setZ(oz)
        this.camera.lookAt(this.controls.target)
    }

    close() {
        if (this.renderer !== null) {
            console.log('Disposing exatomic THREE objects.')
            this.renderer.forceContextLoss()
            this.renderer.dispose()
            this.renderer.setAnimationLoop(null)
            this.renderer = null
        }
    }

    highlight_active(intersects: any[]): void {
        console.log('highlighting active')
    }

    finalizeMouseover(): void {
        const that = this
        this.el.addEventListener(
            'mousemove',
            ((event: MouseEvent): void => {
                event.preventDefault()
                const pos = that.el.getBoundingClientRect()
                const w: number = that.el.offsetWidth || this.model.get('width')
                let h: number = that.el.offsetHeight || this.model.get('height')
                that.mouse.x =  ((event.clientX - pos.x) / w) * 2 - 1
                that.mouse.y = -((event.clientY - pos.y) / h) * 2 + 1
                that.raycaster.setFromCamera(that.mouse, that.camera)
                const intersects = that.raycaster.intersectObjects(that.scene.children)
                if (intersects) {
                    that.highlight_active(intersects)
                }

            }),
            false
        )
    }
}
