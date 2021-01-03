import React from 'react'
import { render } from '@testing-library/react'

import NullScene from './index'

test('NullScene should render', () => {
  const { getByText } = render(<NullScene />)

  expect(getByText('eXatomic')).toBeTruthy()
})
