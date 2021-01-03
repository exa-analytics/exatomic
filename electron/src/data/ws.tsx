import React from 'react'

interface WSState {
    ws: WebSocket | null
    timeout: number
    wsurl: string
    connectInterval: typeof setInterval | Timeout | number | null
    heartbeatInterval: typeof setInterval | Timeout | number | null
}

class WS extends React.Component<null, WSState> {
    state: WSState

    constructor (props: null) {
      /* top-level websocket client interface
        state only supports relevant parameters
        pertaining to the websocket connection
        which has naive reconnect functionality
        */
      super(props)

      this.state = {
        ws: null,
        wsurl: 'ws://localhost:8888/ws',
        timeout: 250,
        connectInterval: null,
        heartbeatInterval: null
        // content: {},
        // data: []
      }

      this.connect = this.connect.bind(this)
      this.checkcon = this.checkcon.bind(this)
      this.heartbeat = this.heartbeat.bind(this)
    }

    componentDidMount (): void {
      this.connect()
    }

    componentWillUnmount (): void {
      const { ws, connectInterval, heartbeatInterval } = this.state
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.onclose = () => { }
        ws.close()
      }
      if (connectInterval) {
        clearTimeout(connectInterval as number)
      }
      if (heartbeatInterval) {
        clearTimeout(heartbeatInterval as number)
      }
      this.setState({ ws: null, connectInterval: null, heartbeatInterval: null })
    }

    heartbeat (): void {
      /* register a heartbeat
        support a heartbeat from the client
        as well as from the server for debugging
        purposes
        */
      const { ws } = this.state
      if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('sending heartbeat')
        ws.send('Ping')
      }
    }

    connect (): void {
      /* establish websocket connection
        additionally will re-establish
        connection after server interruption
        checking every connectInterval seconds,
        which increases on back-to-back failure up
        to 10 seconds
        */
      const ws = new WebSocket(this.state.wsurl)
      const that = this
      const timeout = 250
      ws.onopen = () => {
        console.log('ws client registered')
        if (that.state.connectInterval) {
          clearTimeout(that.state.connectInterval as number)
        }
        const heartbeatInterval = setInterval(function () {
          console.log('sending heartbeat')
          ws.send('heartbeat from client')
        }, 5000)
        this.setState({ ws, timeout, heartbeatInterval })
      }

      ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data)
        console.log('ws client from server', data)
        return false
      }
      ws.onerror = (err) => {
        console.log('ws errored')
        console.log(err)
        ws.close()
      }
      ws.onclose = (evt) => {
        const max = 30
        const mil = 1000
        const backoff = (2 * that.state.timeout) / mil
        const minsec = Math.min(max, backoff)
        console.log('connecting ws in', minsec, 'secs')
        const timeout = 2 * that.state.timeout
        const minmil = Math.min(max * mil, timeout)
        let connectInterval = setTimeout(that.checkcon, minmil)
        that.setState({ timeout, connectInterval })
      }
    }

    checkcon (): void {
      /* simple ready check on the websocket
        */
      const { ws } = this.state
      if (!ws || ws.readyState === WebSocket.CLOSED) this.connect()
    }

    getWsState (key: keyof WSState): any {
      return this.state[key]
    }

    setWsState (key: keyof WSState, value: any) {
      this.setState({ [key]: value })
    }

    render () {
      console.log('ws render')
      // TODO : make icon rotating only if ws is connected
      return (
        <div />
      )
    }
}

export default WS
