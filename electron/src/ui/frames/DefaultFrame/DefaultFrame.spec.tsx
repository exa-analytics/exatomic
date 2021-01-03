import React from 'react'
import { render } from '@testing-library/react'

import DefaultFrame from './index'

test('DefaultFrame should render', () => {
  const { getByText } = render(<DefaultFrame />)

  expect(getByText('eXatomic')).toBeTruthy()
})
