import React from 'react'
import { render } from '@testing-library/react'

import Scene from './index'

test('Scene should render', () => {
  const { getByText } = render(<Scene />)

  expect(getByText('eXatomic')).toBeTruthy()
})
