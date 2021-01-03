import React from 'react'
import { render } from '@testing-library/react'

import Folder from './index'

test('Folder should render', () => {
  const { getByText } = render(<Folder label='Hello' />)

  expect(getByText('Hello')).toBeTruthy()
})
