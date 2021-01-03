import styled, { keyframes } from 'styled-components'

export const Container = styled.div`
    height: 100vh;
    padding: 25px;
    display: flex;
    position: fixed;
    flex-direction: column;
    align-items: center;
    justify-content: center;
`

const rotate = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`

export const Image = styled.img`
    margin: 5px;
    width: 15px;
    animation: ${rotate} 15s linear infinite;
    opacity: 1.0;
`
