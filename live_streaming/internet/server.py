import json

__author__ = 'maeglin89273'


import socket as sk
import thread

HOST = ''
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    print raw_data


def worker_thread(conn, client_addr):
    print "accept a pair of slippers from " + client_addr[0] + ":" + str(client_addr[1])
    try:
        while True:
            json_data = conn.recv(BUFFER_SIZE)
            
            if json_data.isspace() or not json_data:
                break

            raw_data = json.loads(json_data)
            direct_to_model(raw_data)
    finally:
        conn.close()

def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname())
    
    input_stream = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    input_stream.bind((HOST, PORT))
    input_stream.listen(10)

    while True:
        conn, client_addr = input_stream.accept()
        thread.start_new_thread(worker_thread, (conn, client_addr))

    input_stream.close()

if __name__ == '__main__':
    start_server()
