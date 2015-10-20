import socket
import json


def send(device,state):
    HOST, PORT = '127.0.0.1', 4070
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Connect to server and send data
    json_obj=json.dumps({'device': device, 'state': state})
    sock.sendto(json_obj + "\n", (HOST, PORT))
    received = sock.recv(1024)
    print received
    sock.close()
    
if __name__ == '__main__':
    send(1,4)