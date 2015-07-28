import SocketServer
import socket as sk
import json
import sys
import csv
import numpy as np
import pandas as pd
__author__ = 'maeglin89273'

import socket as sk
HOST = '192.168.0.2'
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    bound = 0.4
    global cut_coef
    global slide_size
    global cut_size
    global buffer_length
    global buffer_data
    global buffer_count
    global sent_data
    global sent_count
    global start_recieve
    global first
    global user_counter
    
    if buffer_count < buffer_length:
        print 'collect buffer data'
        buffer_data[buffer_count] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
        buffer_count += 1
    
    else:
        mean = np.mean(buffer_data[:,0])
        now_data = abs(float(raw_data['FFA2']))
        #Over bound and start receive the data
        if ((abs(now_data-mean) > bound) | start_recieve == 1):
            
            
            #Record the data for prediction model
            sent_data[sent_count] = [float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])]
            
            #Record the data index
            sent_count += 1
            start_recieve = 1
            
            if first == 1:
                print "over bound"
                first = 0

            if sent_count == cut_size+1:
                print "finish collecting data"
                out_data = sent_data.T        
                outname = "slipper_data_"+str(user_counter)+".csv"
                out = open("live_data/"+outname, 'wb')
                writer = csv.writer(out)
                writer.writerows(out_data)
                out.close()
                print "New Data " + outname + " finished"

                sent_count = 0
                start_recieve = 0
                first = 1
                user_counter += 1
                
        
            
            #Reach the num of records 
            


class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        json_data = self.request[0]
        raw_data = json.loads(json_data)
        direct_to_model(raw_data)


def start_server(counter):
    print 'current ip address: ' + HOST
    global first 
    global cut_coef
    global cut_size
    global predict_slide_size
    global buffer_length
    global buffer_data
    global buffer_count
    global sent_data
    global sent_count
    global start_recieve
    global user_counter
    cut_coef = 4
    cut_size = 50
    buffer_length = 10
    buffer_data = np.zeros([buffer_length, 6])
    buffer_count = 0
    sent_data = np.zeros([cut_size+1,6])
    sent_count = 1
    start_recieve = 0
    first = 1
    user_counter = counter
    
    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
    server.serve_forever()
    
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "<Usage: <now_user_number>>"
        exit()
    user_counter = int(sys.argv[1])
    start_server(user_counter)
