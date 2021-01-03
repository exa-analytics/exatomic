import React from 'react'
import { Container } from './styles'
import NullScene from './scenes/NullScene'
import DemoScene from './scenes/DemoScene'

export default function SceneManager () {
  const [scene, setScene] = React.useState('demo')

  return (
    <Container>
      {scene === 'demo'
        ? <DemoScene setScene={() => setScene('')} />
        : <NullScene setScene={setScene} />
      }
    </Container>
  )
}
