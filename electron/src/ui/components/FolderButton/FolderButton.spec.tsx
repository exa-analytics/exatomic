import React from 'react'
import { render } from '@testing-library/react'

import FolderButton from './index'

test('FolderButton should render', () => {
  const { getByText } = render(<FolderButton label='Hello' />)

  expect(getByText('Hello')).toBeTruthy()
})
