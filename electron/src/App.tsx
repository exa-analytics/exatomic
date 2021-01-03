import React from 'react'
import { render } from 'react-dom'
import SceneManager from './ui/manager'

const mainElement = document.createElement('div')
mainElement.setAttribute('id', 'root')
document.body.appendChild(mainElement)

const App = () => {
  return (
    <SceneManager />
  )
}

render(<App />, mainElement)
