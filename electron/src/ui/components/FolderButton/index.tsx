import React from 'react'
import ListItem from '@material-ui/core/ListItem'
import Typography from '@material-ui/core/Typography'

export interface FolderButtonProps {
    label: string
    children: React.ReactNode
    onClick(): void
}

export default function FolderButton (props: FolderButtonProps): JSX.Element {
  return (
    <ListItem button key={props.label} onClick={() => props.onClick()}>
      <Typography variant="body2" noWrap>{props.label}</Typography>
      {props.children}
    </ListItem>
  )
}

FolderButton.defaultProps = {
  label: 'Button',
  children: <></>,
  onClick: () => { }
}
