import React from 'react'
import { render } from '@testing-library/react'

import NavFolder from './index'

test('NavFolder should render', () => {
  const { getByText } = render(<NavFolder />)

  expect(getByText('Nav')).toBeTruthy()
})
