import React from 'react'
import { render } from '@testing-library/react'

import Dropdown from './index'

test('Dropdown should render', () => {
  const { getByText } = render(<Dropdown />)

  expect(getByText('Age')).toBeTruthy()
})
