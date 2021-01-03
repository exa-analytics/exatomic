#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import time

from tornado.websocket import websocket_connect

async def main():

    def callback(con):
        print("hit client callback", con)

    def on_message_callback(con):
        print("hit on message callback", con)

    conn = await websocket_connect(
            'ws://localhost:8888/ws',
            callback=callback,
            on_message_callback=on_message_callback,
            ping_interval=1000
        )
    while True:
        print("sending message to server")
        await conn.write_message("things and stuff")
        print("awaiting for read_message")
        msg = await conn.read_message()
        print("client con read message", msg)
        time.sleep(1)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
