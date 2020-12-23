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
// eslint-disable-next-line
import * as util from './util'

const semver = `^${version}`

export class SceneModel extends DOMWidgetModel {
    defaults(): any {
        return {
            ...super.defaults(),
            /* eslint-disable */
            _model_name: 'SceneModel',
            _view_name: 'SceneView',
            _model_module_version: semver,
            _view_module_version: semver,
            _model_module: 'exatomic',
            _view_module: 'exatomic',
            /* eslint-enable */
            width: 200,
            height: 200,
        }
    }
}

interface Meshes {
    scene: any[]
    contour: three.Mesh[]
    frame: three.Mesh[]
    field: three.Mesh[]
    atom: three.Mesh[]
    test: three.Mesh[]
    two: three.Mesh[]
}

// TODO : app interface to SceneView?

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

    prevheight: number

    prevwidth: number

    hudfontsize: number

    hudcanvasdim: number

    promises: Promise<any>

    selected: three.Mesh[]

    meshes: Meshes

    initialize(parameters: any): void {
        super.initialize(parameters)
        this.initListeners()
        this.selected = []
        this.meshes = {
            scene: [],
            contour: [],
            frame: [],
            field: [],
            atom: [],
            test: [],
            two: [],
        }
        this.prevwidth = 0
        this.prevheight = 0
        this.hudfontsize = 28
        this.hudcanvasdim = 1024
        this.promises = this.init()
    }

    init(): Promise<any> {
        /* """
        init
        ---------------
        Promise it will work

        */
        return this.initRenderer().then(() => {
            this.initControls()
            this.initScene()
            this.initObj()
            this.initHud()
            this.finalizeHudcanvas()
            this.finalizeInteractive()
            this.setCameraFromScene()
            this.send({ type: 'init' })
        })
    }

    initScene(): void {
        /* """
        initScene
        ---------------
        A prepped three.js scene containing an ambient light,
        a directional light, and a spot light for shadow creation.

        */
        const faraway = 1000
        const smallnum = 0.3
        const offwhite = 0xdddddd
        const fov = 30
        const aspect = 1
        const near = 1500
        const far = 5000

        this.scene = new three.Scene()
        const ambLight = new three.AmbientLight(offwhite, smallnum)
        const dirLight = new three.DirectionalLight(offwhite, smallnum)
        const sunLight = new three.SpotLight(offwhite, smallnum, 0, Math.PI / 2)
        const shadowcam = new three.PerspectiveCamera(fov, aspect, near, far)

        dirLight.position.set(-faraway, -faraway, -faraway)
        sunLight.position.set(faraway, faraway, faraway)
        sunLight.castShadow = true
        sunLight.shadow = new three.SpotLightShadow(shadowcam)
        sunLight.shadow.bias = 0.0

        this.meshes.scene.push(ambLight)
        this.meshes.scene.push(dirLight)
        this.meshes.scene.push(sunLight)
        this.scene.add(ambLight)
        this.scene.add(dirLight)
        this.scene.add(sunLight)
    }

    initRenderer(): Promise<any> {
        /* """
        initRenderer
        ---------------
        Creates a WebGLRenderer and PerspectiveCamera.

        */
        const fov = 5 // frustum vertical field of view
        const near = 1 // frustum near plane
        const far = 100000 // frustum far plane
        const aspect = 1 // aspect ratio

        const renderer = Promise.resolve(new three.WebGLRenderer({
            antialias: true,
            alpha: true,
        }))
        return renderer.then((ren) => {
            this.renderer = ren
            this.renderer.autoClear = false
            this.renderer.shadowMap.enabled = true
            this.renderer.shadowMap.type = three.PCFSoftShadowMap
            this.el.appendChild(this.renderer.domElement)
            this.camera = new three.PerspectiveCamera(fov, aspect, near, far)
        })
    }

    initControls(): void {
        /* """
        initControls
        ---------------
        Provides mouse input control over the camera in
        the scene by use of a TrackBallControls object.

        */
        if (this.renderer === null) { return }
        this.controls = new TrackBallControls(this.camera, this.renderer.domElement)
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
    }

    initHud(): void {
        /* """
        initHud
        ---------------
        Everything needed to interact with the scene.
        A Raycaster, HUD Scene and OrthographicCamera,
        a mouse and an HTMLCanvas.

        */
        const w = 200
        const h = 200
        this.raycaster = new three.Raycaster()
        this.hudscene = new three.Scene()
        this.mouse = new three.Vector2()
        this.hudcamera = new three.OrthographicCamera(
            -w / 2, w / 2, h / 2, -h / 2, 1, 10,
        )
        this.hudcamera.position.z = 10

        this.hudcanvas = document.createElement('canvas')
        this.hudcanvas.width = this.hudcanvasdim
        this.hudcanvas.height = this.hudcanvasdim
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
        mesh0.position.set(0, 0, -5)
        mesh0.name = 'Icosahedron 0 extra long label that will get cut off eventually around this location'

        const mat1 = new three.MeshBasicMaterial({
            color: 0xFF0000,
            wireframe: true,
        })
        const mesh1 = new three.Mesh(geom, mat1)
        mesh1.position.set(0, 0, 5)
        mesh1.name = 'Icosahedron 1'
        this.scene.add(mesh0)
        this.scene.add(mesh1)
    }

    initListeners(): void {
        /* """
        initListeners
        ---------------
        Register listeners to changes on model

        */
        this.listenTo(this.model, 'change:flag', this.updateFlag)
        this.listenTo(this.model, 'msg:custom', this.handleCustomMsg)
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
        if (this.hudcontext === null) { return }
        this.hudcontext.textAlign = 'left'
        this.hudcontext.textBaseline = 'bottom'
        this.hudcontext.font = `${this.hudfontsize}px Arial`

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
        /* """
        updateFlag
        -----------
        Fired when widget.flag = not widget.flag in the
        notebook. A debugging convenience.

        */
        this.resize()
    }

    resize(): void {
        /* """
        resize
        -----------
        Resizes the renderer and updates all cameras
        and controls to respect the new renderer size.
        Caches previous resize parameters to reduce
        the call stack in the animation loop. Additionally,
        the height of the DOMWidgetView.el element is
        sporadic after a kernel interruption, so an
        explicit check is made for the disconnected state.

        */
        if (this.renderer === null) { return }
        let w, h
        if (this.el.className.includes('disconnected')) {
            w = this.prevwidth || this.model.get('width')
            h = this.prevheight || this.model.get('height')
        } else {
            let pos = this.el.getBoundingClientRect()
            w = Math.floor(pos.width)
            h = Math.floor(pos.height - 5)
            if ((w != this.prevwidth) || (h != this.prevheight)) {
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
                this.prevheight = h
                this.prevwidth = w
            }
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
        if (this.renderer === null) { return }
        this.resize()
        this.controls.update()
        this.renderer.clear()
        this.renderer.render(this.scene, this.camera)
        this.renderer.clearDepth()
        this.renderer.render(this.hudscene, this.hudcamera)
    }

    animate(): void {
        /* """
        animate
        -----------
        Turn on the animation loop.

        */
        if (this.renderer === null) { return }
        this.renderer.setAnimationLoop(this.paint.bind(this))
    }

    handleCustomMsg(msg: any): void {
        /* """
        handleCustomMsg
        -----------------
        Route a message from the kernel.

        */
        if (msg.type === 'close') { this.close() } else { console.log('received msg', msg) }
    }

    setCameraFromScene(): void {
        /* """
        setCameraFromScene
        --------------------
        Find the "center-of-objects" of the scene and point the
        camera towards it from a reasonable distance away from
        all the objects in the scene. Also adjusts scene lighting
        to fit the scene.

        */
        const [, dirLight, sunLight] = this.meshes.scene
        const bbox = new three.Box3().setFromObject(this.scene)
        const { min, max } = bbox
        const ox = (max.x + min.x) / 2
        const oy = (max.y + min.y) / 2
        const oz = (max.z + min.z) / 2
        const px = Math.max(2 * max.x, 60)
        const py = Math.max(2 * max.y, 60)
        const pz = Math.max(2 * max.z, 60)
        const far = Math.max(px, Math.max(py, pz))
        this.camera.position.set(far, far, far)
        this.controls.target.setX(ox)
        this.controls.target.setY(oy)
        this.controls.target.setZ(oz)
        this.camera.lookAt(this.controls.target)
        let mi = Math.min(min.x, Math.min(min.y, min.z))
        let ma = Math.max(max.x, Math.max(max.y, max.z))
        mi = Math.min(-1000, 2 * mi)
        ma = Math.max(1000, 2 * ma)
        dirLight.position.set(mi, mi, mi)
        sunLight.position.set(ma, ma, ma)
    }

    close(): void {
        /* """
        close
        -----------
        Garbage collect all three.js objects when widget
        is closed.

        */
        if (this.renderer === null) { return }
        console.log('disposing contents of Scene')
        // TODO: clear meshes
        //       and all top level attrs
        this.hudtexture.dispose()
        this.renderer.forceContextLoss()
        this.renderer.dispose()
        this.renderer.setAnimationLoop(null)
        this.renderer = null
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
        this.hudsprite.position.set(1000, 1000, 1000)
        if (this.selected.length === 2) {
            const obj0 = this.selected[0]
            const obj1 = this.selected[1]
            const dist = Math.sqrt(
                (obj0.position.x - obj1.position.x) ** 2
                + (obj0.position.y - obj1.position.y) ** 2
                + (obj0.position.z - obj1.position.z) ** 2,
            )
            this.writeHud(`Distance between selected ${dist} units`)
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
        the drawn text are nowhere near each other.

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
        const pad = 6
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
        this.hudsprite.position.set(-width / 2, -height / 2 + this.hudfontsize - pad / 4, 0)
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

    updateHovered(intersects: any[]): void {
        /* """
        updateHovered
        --------------
        Update the HUD based on the current hover-over
        logic. If hovering an object with label information,
        display it. Otherwise fall back to selected
        objects.
        */
        if (!intersects.length) {
            this.writeFromSelected()
            return
        }
        const { name } = intersects[0].object
        // no name no banner
        // TODO: a default "repr"?
        if (!name) { return }
        this.writeHud(name)
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
        information about the object (if available) and the
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
        if (kind === 'mousemove') {
            this.updateHovered(intersects)
        } else if (kind === 'mouseup') {
            this.updateSelected(intersects)
        }
    }

    finalizeInteractive(): void {
        /* """
        finalizeInteractive
        --------------------
        Adds mouse interactivity event listeners

        */
        this.el.addEventListener(
            'mousemove', ((event: MouseEvent): void => {
                this.handleMouseEvent(event, 'mousemove')
            }),
            false,
        )
        this.el.addEventListener(
            'mouseup', ((event: MouseEvent): void => {
                this.handleMouseEvent(event, 'mouseup')
            }),
            false,
        )
    }
}
