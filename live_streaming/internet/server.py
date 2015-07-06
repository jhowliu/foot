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


def worker_thread(conn, client_addr):
    slipper_addr_str = client_addr[0] + ":" + str(client_addr[1])
    print "accept a pair of slippers from " + slipper_addr_str
    try:
        while True:
            json_data = conn.recv(BUFFER_SIZE)

            if is_end(json_data):
                break

            raw_data = json.loads(json_data)
            direct_to_model(raw_data)
    except:
        pass

    finally:
        conn.close()
        print "disconnect the slippers from " + slipper_addr_str

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
