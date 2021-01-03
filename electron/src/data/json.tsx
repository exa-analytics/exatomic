import React from 'react'
import ReactJson from 'react-json-view'
import { useQuery } from 'react-query'

export default function GetData (): React.ReactElement {
  const { isLoading, error, data } = useQuery('repoData', () =>
    fetch('http://localhost:8888/get').then(res => {
      console.log(res.url, res.status)
      return res.json()
    })
  )
  if (isLoading) return <>Loading...</>
  if (error) return <>An error has occurred</> // : {error.message}</>
  return <ReactJson src={data} />
}
