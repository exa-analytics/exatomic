import React from 'react'
import Divider from '@material-ui/core/Divider'

import FolderButton from '../FolderButton'

interface FolderProps {
    label: string
    children: React.ReactNode
    initialValue: boolean
}

export default function Folder (props: FolderProps): JSX.Element {
  const [folderOpen, setFolderOpen] = React.useState(props.initialValue)

  const handleOpen = () => {
    if (folderOpen) {
      setFolderOpen(false)
    } else {
      setFolderOpen(true)
    }
  }

  return (
    <>
      <FolderButton label={props.label} onClick={handleOpen} />
      {folderOpen
        ? <><Divider />{props.children}<Divider /></> : null
      }
    </>
  )
}

Folder.defaultProps = {
  label: 'Folder',
  children: <></>,
  initialValue: false
}
