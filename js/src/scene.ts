// Copright (c) 2015-2020, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/* """
=================
scene.ts
=================
A 3D scene for exatomic

*/

import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base'

import * as three from 'three'
import * as TrackBallControls from 'three-trackballcontrols'
import { version } from '../package.json'
import * as util from './util'

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

    hudfontsize: number

    promises: Promise<any>

    selected: three.Mesh[]

    initialize(parameters: any): void {
        super.initialize(parameters)
        this.initListeners()
        this.selected = []
        this.hudfontsize = 28
        this.promises = this.init()
        this.displayed.then(() => {
            this.resize()
            this.setCameraFromScene()
        })
    }

    init(): Promise<any> {
        /* """
        init
        ---------------
        Promise it will work
        */
        window.addEventListener('resize', this.resize.bind(this))
        return Promise.all([
            this.initScene(),
            this.initRenderer(),
            this.initHud(),
        ]).then(() => {
            this.initObj()
            this.finalizeHudcanvas()
            this.finalizeInteractive()
        })
    }

    initScene(): any {
        /* """
        initScene
        ---------------
        An opinionated three.js Scene
        */
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
        // })
        sunlight.shadow.bias = 0.0
        scene.add(amlight)
        scene.add(dlight0)
        scene.add(sunlight)
        return Promise.resolve(scene).then((scn) => {
            this.scene = scn
        })
    }

    initRenderer(): any {
        /* """
        initRenderer
        ---------------
        A WebGLRenderer, PerspectiveCamera and
        TrackBallControls object.
        */
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
        /* """
        initHud
        ---------------
        Everything needed to interact with the scene.
        A Raycaster, HUD Scene and OrthographicCamera,
        a mouse and an HTMLCanvas.
        */
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
                1, 10,
            )).then((camera) => {
                this.hudcamera = camera
                this.hudcamera.position.z = 10
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
        /* """
        initObj
        ------------
        Create some test meshes to test hud and interactive
        functionality.

        */
        const geom = new three.IcosahedronGeometry(2, 1)
        const mat0 = new three.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true,
        })
        const mesh0 = new three.Mesh(geom, mat0)
        mesh0.position.set(0, 0, -3)
        mesh0.name = 'Icosahedron 0 Extra Long Name Probably a Problem And More Long Extra Description'

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
    }

    finalizeHudcanvas(): void {
        /* """
        finalizeHudcanvas
        ------------------
        Set up everything needed to write text to
        the hudcanvas and display it in the same
        renderer.

        */
        if (this.renderer === null) { return }
        this.hudcontext = this.hudcanvas.getContext('2d')
        if (this.hudcontext !== null) {
            this.hudcontext.textAlign = 'left'
            this.hudcontext.textBaseline = 'bottom'
            this.hudcontext.font = `${this.hudfontsize}px Arial`
        }
        this.hudtexture = new three.Texture(this.hudcanvas)
        this.hudtexture.anisotropy = this.renderer.capabilities.getMaxAnisotropy()
        this.hudtexture.minFilter = three.NearestMipMapLinearFilter
        this.hudtexture.magFilter = three.NearestFilter
        this.hudtexture.needsUpdate = true
        const material = new three.SpriteMaterial({ map: this.hudtexture })
        this.hudsprite = new three.Sprite(material)
        this.hudsprite.position.set(1000, 1000, 1000)
        this.hudsprite.scale.set(512, 512, 1)
        this.hudscene.add(this.hudcamera)
        this.hudscene.add(this.hudsprite)
    }

    updateFlag(): void {
        this.resize()
    }

    resize(): void {
        /* """
        resize
        -----------
        Resizes the renderer and updates all cameras
        and controls to respect the new renderer size.

        */
        const w: number = this.el.offsetWidth || this.model.get('width')
        let h: number = this.el.offsetHeight || this.model.get('height')
        // hack canvas in el is 6 smaller than el
        if (h === this.el.offsetHeight) { h -= 6 }
        if (this.renderer !== null) {
            this.renderer.setSize(w, h)
            this.camera.aspect = w / h
            this.camera.updateProjectionMatrix()
            this.camera.updateMatrix()
            this.hudcamera.left = -w / 2
            this.hudcamera.right = w / 2
            this.hudcamera.top = h / 2
            this.hudcamera.bottom = -h / 2
            this.hudcamera.updateProjectionMatrix()
            this.hudcamera.updateMatrix()
            this.controls.handleResize()
        }
    }

    render(): any {
        /* """
        render
        -----------
        Return all promises

        */
        return this.promises.then(this.animate.bind(this))
    }

    paint(): void {
        /* """
        paint
        -----------
        Update the renderer and render both the scene
        and hudscene.

        */
        if (this.renderer !== null) {
            // TODO : there must be a way to not call resize
            this.resize()
            this.controls.update()
            this.renderer.clear()
            this.renderer.render(this.scene, this.camera)
            this.renderer.clearDepth()
            this.renderer.render(this.hudscene, this.hudcamera)
        }
    }

    animate(): void {
        /* """
        animate
        -----------
        Turn on the animation loop.

        */
        if (this.renderer !== null) {
            this.renderer.setAnimationLoop(this.paint.bind(this))
        }
    }

    setCameraFromScene(): void {
        /* """
        setCameraFromScene
        --------------------
        Find the "center-of-objects" of the scene and point the
        camera towards it from a reasonable distance away from
        all the objects in the scene.

        */
        const bbox = new three.Box3().setFromObject(this.scene)
        const { min } = bbox
        const { max } = bbox
        const ox = (max.x + min.x) / 2
        const oy = (max.y + min.y) / 2
        const oz = (max.z + min.z) / 2
        const px = Math.max(2 * max.x, 30)
        const py = Math.max(2 * max.y, 30)
        const pz = Math.max(2 * max.z, 30)
        this.camera.position.set(px, py, pz)
        this.controls.target.setX(ox)
        this.controls.target.setY(oy)
        this.controls.target.setZ(oz)
        this.camera.lookAt(this.controls.target)
    }

    close(): void {
        /* """
        close
        -----------
        Garbage collect everything on this.

        */
        if (this.renderer !== null) {
            // console.log('Disposing exatomic THREE objects.')
            this.renderer.forceContextLoss()
            this.renderer.dispose()
            this.renderer.setAnimationLoop(null)
            this.renderer = null
        }
    }

    writeFromSelected(): void {
        /* """
        writeFromSelected
        ---------------------
        Based on the number (and kind) of objects selected,
        when no other banner is being displayed, display the
        calculable information based on what was selected.

        For example, when two Meshes are selected with positions,
        we can compute the distance between them and write out
        the result when both objects are selected and no other
        objects are hovered.

        */

        // TODO: what to do when given things are selected
        if (this.selected.length === 2) {
            const obj0 = this.selected[0]
            const obj1 = this.selected[1]
            const dist = Math.sqrt(
                (obj0.position.x - obj1.position.x) ** 2
                + (obj0.position.y - obj1.position.y) ** 2
                + (obj0.position.z - obj1.position.z) ** 2,
            )
            this.writeHud(`Distance ${dist} units`)
        }
    }

    writeHud(banner: string): void {
        /* """
        writeHud
        ---------------
        Write into a CanvasRenderingContext2D the given banner.
        The hudcontext API generally accepts (x, y, w, h) and
        has coordinates laid out as follows:

                y=0
            x=0 +---+ x=w -+ 1024
                |   |      |
                +---+      |
                y=h        |
                |          |
                +----------+
                1024

        All based on the hudcanvas (1024x1024). By measuring
        the text width and knowing the font size, construct a
        bordered text box to frame the text banner. Then write
        the text to the hudcontext. Currently only write one line
        of text, as much that will fit into the full hudcanvas size
        of 1024 pixels.

        The hudcontext is where we do the drawing, but the sprite
        displays the result. The hudcanvas (housing the hudcontext)
        was the source for the three.Texture that is the material
        the sprite renders.

        The sprite is rendered in a hudscene using an orthographic
        camera. Since we only write one line of text into an otherwise
        square hudcanvas, the center of the sprite and the center of
        the painted text are nowhere near each other.

        Reassign the center of the sprite to be the top left corner
        of the hudcontext using the following coordinate system:

            y
            1 - - - 1
            |_____| |
            |       |
            0 - - - 1 x

        So that the apparent "center" of the sprite will be closer
        to the actual drawn text. Finally, position the sprite (in
        orthographic coordinates [-w/2, w/2, -h/2, h/2]).
        */

        if (this.hudcontext === null) {
            return
        }
        this.hudcontext.clearRect(0, 0, this.hudcanvas.width, this.hudcanvas.height)

        // frame the text
        const pad = 8
        const textWidth = this.hudcontext.measureText(banner).width

        // with a black border
        this.hudcontext.fillStyle = 'rgba(0,0,0,0.9)'
        this.hudcontext.fillRect(0, 0, textWidth + 2 * pad, this.hudfontsize + 2 * pad)

        // and light background
        this.hudcontext.fillStyle = 'rgba(245,245,245,0.9)'
        this.hudcontext.fillRect(
            pad / 2, pad / 2, textWidth + pad, this.hudfontsize + pad,
        )
        // for easy to read black text
        this.hudcontext.fillStyle = 'rgba(0,0,0,0.95)'
        this.hudcontext.fillText(banner, pad, this.hudfontsize + pad)

        this.hudsprite.center.x = 0
        this.hudsprite.center.y = 1

        const width = this.el.offsetWidth
        const height = this.el.offsetHeight
        this.hudsprite.position.set(-width / 2, -height / 2 + this.hudfontsize, 0)
        this.hudsprite.material.needsUpdate = true
        this.hudtexture.needsUpdate = true
    }

    updateSelected(intersects: any[]): void {
        /* """
        updateSelected
        ----------------
        If the first element of intersects is not selected,
        highlight it and add it to selected objects.
        Otherwise, if the first element of intersects is
        selected, unhighlight it and remove it from
        selected objects
        */
        if (!intersects.length) { return }
        const obj = intersects[0].object
        const uuids = this.selected.map((o) => o.uuid)
        const { uuid } = obj
        const idx = uuids.indexOf(uuid)
        if (idx > -1) {
            obj.material.color.setHex(obj.oldHex)
            this.selected.splice(idx, 1)
        } else {
            obj.oldHex = obj.material.color.getHex()
            const newHex = util.lightenColor(obj.oldHex)
            obj.material.color.setHex(newHex)
            this.selected.push(obj)
        }
    }

    handleMouseEvent(event: MouseEvent, kind: string): void {
        /* """
        handleMouseEvent
        ------------------
        Compute the mouse position in 2D hovering over
        the 3D scene. Cast a ray from that 2D position
        into the 3D scene and get all intersecting objects.

        There are two distinct interactions with the three.js
        scene, hover-over and selection. The hover-over displays
        information about the object (if provided) and the
        selection allows to select multiple items and compute
        values related to collections of objects in the scene.
        */
        event.preventDefault()
        const pos = this.el.getBoundingClientRect()
        const w: number = this.el.offsetWidth
        const h: number = this.el.offsetHeight
        this.mouse.x = ((event.clientX - pos.x) / w) * 2 - 1
        this.mouse.y = -((event.clientY - pos.y) / h) * 2 + 1
        this.raycaster.setFromCamera(this.mouse, this.camera)
        const intersects = this.raycaster.intersectObjects(this.scene.children)
        if (kind === 'mouseup') {
            this.updateSelected(intersects)
        } else if (kind === 'mousemove') {
            // update the hudscene with the label for the
            // currently hovered over object.
            // fall back to banner based on selected
            if (!intersects.length) {
                this.writeFromSelected()
                return
            }
            const obj = intersects[0].object
            // no name no banner
            if (!obj.name) { return }
            this.writeHud(obj.name)
        }
    }

    finalizeInteractive(): void {
        /* """
        finalizeInteractive
        --------------------
        Adds mouse interactivity event listeners

        */
        this.el.addEventListener(
            'mousemove',
            ((event: MouseEvent): void => {
                this.handleMouseEvent(event, 'mousemove')
            }),
            false,
        )
        this.el.addEventListener(
            'mouseup',
            ((event: MouseEvent): void => {
                this.handleMouseEvent(event, 'mouseup')
            }),
            false,
        )
    }
}
