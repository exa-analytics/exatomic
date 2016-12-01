#!/usr/bin/python

import datetime
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
from tornado import gen
from tornado.options import define, parse_command_line


class WSHandler(tornado.websocket.WebSocketHandler):
    clients = []
    def open(self):
        print('new connection')
        self.write_message("Hello World")
        WSHandler.clients.append(self)

    def on_message(self, message):
        print('message received %s' % message)
        self.write_message('ECHO: ' + message)

    def on_close(self):
        print('connection closed')
        WSHandler.clients.remove(self)

    @classmethod
    def write_to_clients(cls):
        print("Writing to clients")
        for client in cls.clients:
            client.write_message("Hi there!")

    def check_origin(self, origin):
        return True



application = tornado.web.Application([
  (r'/websocket', WSHandler),
])

@gen.coroutine
def client_main():
    conn = yield tornado.websocket.websocket_connect('ws://localhost:8001/websocket')
    print((yield conn.read_message()))
    yield conn.write_message('hello')
    print((yield conn.read_message()))

if __name__ == "__main__":
    define('ioloop', type=str, default=None,
           callback=tornado.ioloop.IOLoop.configure)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8001)
    tornado.ioloop.IOLoop.instance().run_sync(client_main)
