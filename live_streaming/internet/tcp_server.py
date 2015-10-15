import SocketServer
import json

__author__ = 'maeglin89273'


import socket as sk
import thread

HOST = ''
PORT = 3070

BUFFER_SIZE = 1024

def direct_to_model(raw_data):
    print raw_data['FFA2']

class TCPHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        self.slipper_addr_str = self.client_address[0] + ':' + str(self.client_address[1])
        print 'accept a pair of slippers from ' + self.slipper_addr_str

    def handle(self):
        while True:
            json_data = self.request.recv(BUFFER_SIZE)

            try:
                raw_data = json.loads(json_data)
                direct_to_model(raw_data)
            except ValueError:
                print json_data
                print ValueError

    def finish(self):
        print 'disconnect the slippers from ' + self.slipper_addr_str

def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname())

    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()

if __name__ == '__main__':
    start_server()
