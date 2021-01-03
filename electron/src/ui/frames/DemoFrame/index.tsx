import React from 'react'
import List from '@material-ui/core/List'
import ListItem from '@material-ui/core/ListItem'
import Typography from '@material-ui/core/Typography'

import Header from '../../components/Header'
import Folder from '../../components/Folder'
import Dropdown from '../../components/Dropdown'
import JsonView from '../../components/JsonView'
import FolderSlider from '../../components/FolderSlider'
import FolderButton from '../../components/FolderButton'

import WS from '../../../data/ws'
import { Image } from '../../styles'
import NavFolder, { NavFolderProps } from '../../components/NavFolder'

export interface DemoFrameProps extends NavFolderProps {
    label: string
    addCube(): void
    delCube(): void
    color: string
    setColor(color: string): void
    opacity: number
    setOpacity(opacity: number | number[]): void
}

export default function DemoFrame (props: DemoFrameProps) {
  return (
    <Header label={props.label}>
      <List>
        <NavFolder setScene={props.setScene} />
        <Folder label='Edit'>
          <FolderButton label='Add Cube' onClick={() => props.addCube()} />
          <FolderButton label='Remove Cube' onClick={() => props.delCube()} />
          <FolderSlider
            label='Opacity'
            min={0} max={1} step={0.01}
            value={props.opacity}
            setValue={props.setOpacity}
          />
          <Dropdown
            label='Color'
            options={['red', 'green', 'blue']}
            value={props.color}
            setValue={props.setColor}
          />
        </Folder>
        <Folder label='Comms'>
          <JsonView />
          <FolderButton label='Websocket' onClick={() => console.log('ws')}>
            <WS />
            <Image
              src="https://www.vectorlogo.zone/logos/reactjs/reactjs-icon.svg"
              alt="ReactJS logo"
            />
          </FolderButton>
        </Folder>
      </List>
    </Header>
  )
}

// TODO : defaultProps required vs. optional
DemoFrame.defaultProps = {
  label: 'Demo',
  color: 'red',
  opacity: 1.0,
  setColor: (color: string) => { },
  setOpacity: (opacity: number) => { },
  addCube: () => { },
  delCube: () => { },
  ...NavFolder.defaultProps
}
