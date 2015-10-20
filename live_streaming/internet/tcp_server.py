import SocketServer
import json

__author__ = 'maeglin89273'

import socket as sk
import thread
import sys

HOST = ''
PORT = 3070

BUFFER_SIZE = 512
if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
    out = open(FILENAME, 'a')
else:
    exit()

out.write("Axis1,Axis2,Axis3,Axis4,Axis5,Axis6,Label,Timestamp")

def direct_to_model(raw_data):
    out.write(",".join([raw_data['FFA2'], raw_data['Label'], raw_data['Timestamp']]) + '\n')
    print raw_data

class TCPHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        self.data = []
        self.buffer = ""
        self.slipper_addr_str = self.client_address[0] + ":" +str(self.client_address[1])
        print "Connect the slippers from " + self.slipper_addr_str

    def handle(self):
        while True:
            json_data = self.request.recv(BUFFER_SIZE)
            if not json_data:
                break
            self.parse(json_data)
            map(lambda x: direct_to_model(x), self.data)
            # Clean up data list
            self.data = []

    def parse(self, json_data):
        try:
            self.data.append(json.loads(json_data))
        except:
            tmp = self.buffer + json_data
            self.buffer = ""
            for x in tmp.replace("}{", "}||{").split("||"):
                try:
                    self.data.append(json.loads(string))
                except:
                    self.buffer = self.buffer + x

    def finish(self):
        print 'disconnect the slippers from ' + self.slipper_addr_str
        out.close()

def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname())
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()

if __name__ == '__main__':
    start_server()
