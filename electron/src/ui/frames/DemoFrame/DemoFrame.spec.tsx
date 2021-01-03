import React from 'react'
import { render } from '@testing-library/react'

import DemoFrame from './index'

test('DemoFrame should render', () => {
  const { getByText } = render(<DemoFrame />)

  expect(getByText('eXatomic')).toBeTruthy()
})
