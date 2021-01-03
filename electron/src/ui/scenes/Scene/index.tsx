import React from 'react'
import * as THREE from 'three'
import { TrackballControls } from 'three-trackballcontrols-ts'
import Toolbar from '@material-ui/core/Toolbar'

export interface SceneProps {
    scene: THREE.Scene
    renderer: THREE.WebGLRenderer
}

export interface SceneState {
    camera: THREE.PerspectiveCamera
    controls: TrackballControls | null
}

export default class Scene extends React.Component<SceneProps, SceneState> {
    static defaultProps: SceneProps
    ref: React.RefObject<HTMLDivElement>

    constructor (props: SceneProps) {
      super(props)
      this.ref = React.createRef()
      this.state = {
        camera: new THREE.PerspectiveCamera(75, 1, 0.1, 1000),
        controls: null
      }
    }

    resize (): void {
      const w = window.innerWidth
      const h = window.innerHeight
      const { renderer } = this.props
      const { camera, controls } = this.state
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      camera.updateMatrix()
      renderer.setSize(w, h)
        controls?.handleResize()
    }

    paint (): void {
      for (const cube of this.props.scene.children) {
        cube.rotation.x += 0.01
        cube.rotation.y += 0.01
      }
      const { scene, renderer } = this.props
      const { camera, controls } = this.state
        controls?.update()
        renderer.clear()
        renderer.render(scene, camera)
    }

    componentDidMount (): void {
      const { renderer } = this.props
      let { camera, controls } = this.state
        this.ref.current?.appendChild(renderer.domElement)
        controls = new TrackballControls(camera, renderer.domElement)
        controls.staticMoving = true
        controls.rotateSpeed = 10.0
        controls.zoomSpeed = 5.0
        controls.panSpeed = 0.5
        renderer.autoClear = false
        renderer.shadowMap.enabled = true
        renderer.shadowMap.type = THREE.PCFSoftShadowMap
        window.addEventListener('resize', () => this.resize())
        renderer.setAnimationLoop(() => this.paint())
        camera.position.z = 20
        this.setState({ controls })
        this.resize()
    }

    componentWillUnmount (): void {
      const { renderer } = this.props
      // renderer.forceContextLoss()
      // renderer.dispose()
      renderer.setAnimationLoop(null)
    }

    render (): React.ReactComponent {
      return (
        <main>
          <div>
            <Toolbar />
            <div ref={this.ref} />
          </div>
        </main>
      )
    }
}

Scene.defaultProps = {
  scene: new THREE.Scene(),
  renderer: new THREE.WebGLRenderer({ antialias: true, alpha: true })
}
