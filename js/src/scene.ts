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

import * as util from './util'

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

    raycaster: three.Raycaster

    mouse: three.Vector2

    hudscene: three.Scene

    hudcamera: three.OrthographicCamera

    hudcanvas: HTMLCanvasElement

    hudcontext: CanvasRenderingContext2D | null

    hudtexture: three.Texture

    hudsprite: three.Sprite

    promises: Promise<any>

    selected: three.Mesh[]

    initialize(parameters: any): void {
        super.initialize(parameters)
        this.initListeners()
        this.selected = []
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
            this.finalizeHudcanvas()
            this.finalizeMouseover()
            this.finalizeMouseup()
            // TODO: why this no fire?
            window.dispatchEvent(new Event('resize'))
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

    finalizeHudcanvas(): void {
        if (this.renderer === null) { return }
        this.hudcontext = this.hudcanvas.getContext('2d')
        if (this.hudcontext !== null) {
            this.hudcontext.textAlign = 'left'
            this.hudcontext.textBaseline = 'bottom'
            this.hudcontext.font = '64px Arial'
        }
        this.hudtexture = new three.Texture(this.hudcanvas)
        this.hudtexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy()
        this.hudtexture.minFilter = three.NearestMipMapLinearFilter
        this.hudtexture.magFilter = three.NearestFilter
        this.hudtexture.needsUpdate = true
        const material = new three.SpriteMaterial({ map: this.hudtexture })
        this.hudsprite = new three.Sprite(material)
        this.hudsprite.position.set(1000, 1000, 1000)
        this.hudsprite.scale.set(256, 256, 1)
        this.hudscene.add(this.hudsprite)
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

    updateHud(intersects: any[]): void {
        // update the hudscene with the label for the
        // currently hovered over object.
        // TODO : look into the tooltip machinery
        if (this.hudcontext === null) { return }
        // wipe the canvas
        this.hudcontext.clearRect(0, 0,
                                  this.hudcanvas.width,
                                  this.hudcanvas.height)
        // make sure we are hovering over an object
        if (!intersects.length) {
            this.hudsprite.position.set(1000, 1000, 1000)
            return
        }
        const obj = intersects[0].object
        if (!obj.name) { return }

//        this.hudcontext.fillStyle = 'rgba(245,245,245,0.9)'
//        let w = this.hudcontext.measureText(obj.name).width
//        let h = 72
//        console.log("fill rect (x, y, w, h)", 0, 0, w + 8, h)
//        this.hudcontext.fillRect(0, 0, w + 8, 72)
//        this.hudcontext.fillStyle = 'rgba(0,0,0,0.95)'
//        console.log("fill text (w, 512, 512)", w, 0, 0)
//        this.hudcontext.fillText(obj.name, 0, 0)
//        w = this.el.offsetWidth
//        h = this.el.offsetHeight - 6
//        console.log("hudsprite position", 0, 0, 1)
//        this.hudsprite.position.set(0, 0, 1)
//        this.hudsprite.material.needsUpdate = true
//        this.hudtexture.needsUpdate = true

        // this is not quite aligned
        this.hudcontext.fillStyle = 'rgba(245,245,245,0.9)'
        let w = this.hudcontext.measureText(obj.name).width
        console.log("fill rect (x, y, w, h)", 512 - 2, 512 - 60, w + 6, 72)
        this.hudcontext.fillRect(512 - 2, 512 - 60, w + 6, 72)
        this.hudcontext.fillStyle = 'rgba(0,0,0,0.95)'
        console.log("fill text (w, 512, 512)", w, 512, 512)
        this.hudcontext.fillText(obj.name, 512, 512)
        w = this.el.offsetWidth
        const h = this.el.offsetHeight - 6
        console.log("hudsprite position", -w / 2 + 2, -h / 2 + 4, 1)
        this.hudsprite.position.set(-w / 2 + 2, - h / 2 + 4, 1)
        this.hudsprite.material.needsUpdate = true
        this.hudtexture.needsUpdate = true
    }

    finalizeMouseover(): void {
        const that = this
        this.el.addEventListener(
            'mousemove',
            ((event: MouseEvent): void => {
                event.preventDefault()
                const pos = that.el.getBoundingClientRect()
                const w: number = that.el.offsetWidth
                const h: number = that.el.offsetHeight
                that.mouse.x =  ((event.clientX - pos.x) / w) * 2 - 1
                that.mouse.y = -((event.clientY - pos.y) / h) * 2 + 1
                that.raycaster.setFromCamera(that.mouse, that.camera)
                const intersects = that.raycaster.intersectObjects(that.scene.children)
                that.updateHud(intersects)
            }),
            false
        )
    }

    updateActive(intersects: any[]): void {
        // if first element of intersects is not in selected,
        // highlight it and add it to selected.
        // otherwise, if first element of intersects is in
        // selected, unhighlight it and remove it from
        // selected
        if (!intersects.length) { return }
        const obj = intersects[0].object
        const uuids = this.selected.map(obj => obj.uuid)
        const uuid = obj.uuid
        const idx = uuids.indexOf(uuid)
        if (idx > -1) {
            obj.material.color.setHex(obj.oldHex)
            this.selected.splice(idx, 1)
        } else {
            obj.oldHex = obj.material.color.getHex()
            let newHex = util.lightenColor(obj.oldHex)
            obj.material.color.setHex(newHex)
            this.selected.push(obj)
        }
    }

    finalizeMouseup(): void {
        const that = this
        this.el.addEventListener(
            'mouseup',
            ((event: MouseEvent): void => {
                event.preventDefault()
                const pos = that.el.getBoundingClientRect()
                const w: number = that.el.offsetWidth
                const h: number = that.el.offsetHeight
                that.mouse.x =  ((event.clientX - pos.x) / w) * 2 - 1
                that.mouse.y = -((event.clientY - pos.y) / h) * 2 + 1
                // console.log('pos, w, h, mouse', pos, w, h, that.mouse)
                that.raycaster.setFromCamera(that.mouse, that.camera)
                const intersects = that.raycaster.intersectObjects(that.scene.children)
                that.updateActive(intersects)
            }),
            false
        )
    }
}
