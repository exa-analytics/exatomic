import React from 'react'
import { render } from '@testing-library/react'

import Header from './index'

test('Header should render', () => {
  const { getByText } = render(<Header label='Hello' />)

  expect(getByText('eXatomic:Hello')).toBeTruthy()
})
