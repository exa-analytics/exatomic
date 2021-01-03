import React from 'react'

import DefaultFrame from '../../frames/DefaultFrame'
import Toolbar from '@material-ui/core/Toolbar'

interface NullSceneProps {
    children: React.ReactNode
    setScene(scene: string): void
}

export default function NullScene (props: NullSceneProps) {
  return (
    <>
      <DefaultFrame
        setScene={props.setScene}
      />
      <main>
        <div>
          <Toolbar />
          {props.children}
        </div>
      </main>
    </>
  )
}

NullScene.defaultProps = {
  setScene (scene: string): void { },
  children: <></>
}
