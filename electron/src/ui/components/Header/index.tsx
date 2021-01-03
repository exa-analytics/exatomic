import React from 'react'
import AppBar from '@material-ui/core/AppBar'
import Drawer from '@material-ui/core/Drawer'
import Divider from '@material-ui/core/Divider'
import Toolbar from '@material-ui/core/Toolbar'
import MenuIcon from '@material-ui/icons/Menu'
import IconButton from '@material-ui/core/IconButton'
import Typography from '@material-ui/core/Typography'
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'
import { makeStyles } from '@material-ui/core/styles'

const useStyles = makeStyles({
  root: {
    minWidth: 282
  }
})

interface HeaderProps {
    label: string
    children: React.ReactNode
}

export default function Header (props: HeaderProps): React.ReactNode {
  const classes = useStyles()
  const [guiOpen, setGuiOpen] = React.useState(false)
  const coreName = 'eXatomic'
  let longName = coreName
  if (props.label) {
    longName += ':' + props.label
  }

  return (
    <>
      <AppBar
        position='fixed'
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={() => setGuiOpen(true)}
            edge="start"
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap>
            {longName}
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="persistent"
        anchor="left"
        open={guiOpen}
      >
        <div className={classes.root}>
          <Toolbar>
            <IconButton onClick={() => setGuiOpen(false)}>
              <ChevronLeftIcon />
              <Typography align="center" variant="h6" noWrap>
                {coreName}
              </Typography>
            </IconButton>
          </Toolbar>
        </div>
        <Divider />
        {props.children}
      </Drawer>
    </>
  )
}

Header.defaultProps = {
  label: 'Header',
  children: <></>
}
