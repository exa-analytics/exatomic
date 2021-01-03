import React from 'react'
import { render } from '@testing-library/react'

import FolderSlider from './index'

test('FolderSlider should render', () => {
  const { getByText } = render(<FolderSlider />)

  expect(getByText('Volume')).toBeTruthy()
})
