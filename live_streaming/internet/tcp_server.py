import SocketServer
import json

__author__ = 'maeglin89273'


import socket as sk
import thread

HOST = ''
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    print raw_data

def is_end(json_data):
    #empty string OR eot, end of transimisson OR spaces
    return not json_data or \
    (len(json_data) == 1 and ord(json_data) == 4) or \
    json_data.isspace()

class TCPHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        self.slipper_addr_str = self.client_address[0] + ':' + str(self.client_address[1])
        print 'accept a pair of slippers from ' + self.slipper_addr_str

    def handle(self):
        while True:
            json_data = self.request.recv(BUFFER_SIZE)
            if is_end(json_data):
                break

            raw_data = json.loads(json_data)
            direct_to_model(raw_data)

    def finish(self):
        print 'disconnect the slippers from ' + self.slipper_addr_str

def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname())
    
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()

if __name__ == '__main__':
    start_server()
