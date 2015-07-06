import SocketServer
import json

__author__ = 'maeglin89273'


import socket as sk


HOST = ''
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    print raw_data


class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        json_data = self.request[0]
        raw_data = json.loads(json_data)
        direct_to_model(raw_data)


def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname())
    
    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
    server.serve_forever()


if __name__ == '__main__':
    start_server()
