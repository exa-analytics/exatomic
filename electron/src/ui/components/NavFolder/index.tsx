import React from 'react'
import ListItem from '@material-ui/core/ListItem'
import Typography from '@material-ui/core/Typography'
import Folder from '../Folder'

import FolderItem from '../FolderButton'

export interface NavFolderProps {
    setScene(scene: string): void
}

export default function NavFolder (props: NavFolderProps) {
  const label = 'Back'
  return (
    <Folder label='Nav'>
      <ListItem button key={label} onClick={() => props.setScene('')}>
        <Typography variant="body2" noWrap>{label}</Typography>
      </ListItem>
    </Folder>
  )
}

NavFolder.defaultProps = {
  setScene: (scene: string) => { }
}
