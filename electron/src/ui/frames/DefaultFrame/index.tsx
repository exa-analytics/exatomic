import React from 'react'
import Header from '../../components/Header'
import Folder from '../../components/Folder'
import FolderButton from '../../components/FolderButton'

interface DefaultFrameProps {
    label: string
    setScene(scene: string): void
}

export default function DefaultFrame (props: DefaultFrameProps) {
  return (
    <Header label={props.label}>
      <Folder label='Launch'>
        <FolderButton label={'Demo'} onClick={() => props.setScene('demo')} />
      </Folder>
    </Header>
  )
}

DefaultFrame.defaultProps = {
  label: 'Home',
  setScene: (scene: string) => { }
}
