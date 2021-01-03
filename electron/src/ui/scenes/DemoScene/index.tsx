import React from 'react'
import * as THREE from 'three'
import Scene, { SceneProps } from '../Scene'
import DemoFrame, { DemoFrameProps } from '../../frames/DemoFrame'

export interface DemoSceneProps extends SceneProps, DemoFrameProps {
}

export interface DemoSceneState {
    geometry: THREE.BoxGeometry
    material: THREE.MeshBasicMaterial
    opacity: number
    color: string
}

export default class DemoScene extends React.Component<DemoSceneProps, DemoSceneState> {
    static defaultProps: DemoSceneProps

    constructor (props: DemoSceneProps) {
      super(props)
      this.state = {
        opacity: 0.5,
        color: 'green',
        geometry: new THREE.BoxGeometry(1, 1, 1),
        material: new THREE.MeshBasicMaterial({
          transparent: true
        })
      }
      this.setColor = this.setColor.bind(this)
      this.setOpacity = this.setOpacity.bind(this)
    }

    componentDidMount (): void {
      const { color, opacity, material } = this.state
      material.color.set(color)
      material.opacity = opacity
      material.needsUpdate = true
    }

    delCube (): void {
      if (this.props.scene.children) {
        this.props.scene.remove(this.props.scene.children[0])
      }
    }

    addCube (): void {
      const d = 20
      const x = Math.floor((Math.random() - 0.5) * d)
      const y = Math.floor((Math.random() - 0.5) * d)
      const z = Math.floor((Math.random() - 0.5) * d)
      const cube = new THREE.Mesh(this.state.geometry, this.state.material)
      cube.position.x = x
      cube.position.y = y
      cube.position.z = z
      this.props.scene.add(cube)
    }

    setColor (color: string): void {
      const { material } = this.state
      material.color.set(color)
      this.setState({ color: color })
    }

    setOpacity (opacity: number): void {
      const { material } = this.state
      material.opacity = opacity
      material.needsUpdate = true
      this.setState({ opacity: opacity })
    }

    render () {
      return (
        <>
          <DemoFrame
            setScene={() => this.props.setScene('')}
            addCube={() => this.addCube()}
            delCube={() => this.delCube()}
            color={this.state.color}
            setColor={this.setColor}
            opacity={this.state.opacity}
            setOpacity={this.setOpacity}
          />
          <Scene {...this.props} />
        </>
      )
    }
}

DemoScene.defaultProps = {
  ...Scene.defaultProps,
  ...DemoFrame.defaultProps
}
