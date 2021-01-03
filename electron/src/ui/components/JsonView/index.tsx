import React from 'react'
import { QueryClient, QueryClientProvider } from 'react-query'

import Folder from '../Folder'
import GetData from '../../../data/json'

const queryClient = new QueryClient()

export default function JsonView () {
  return (
    <QueryClientProvider client={queryClient}>
      <Folder label='Json'>
        <GetData />
      </Folder>
    </QueryClientProvider>
  )
}
