import SocketServer
import json
import sys
import numpy as np
import pandas as pd
__author__ = 'maeglin89273'

import socket as sk
sys.path.append('~/Work/foot/')
import train_dtw as train
HOST = '192.168.0.2'
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    bound = 0.6
    buffer_length = 5
    global buffer_data
    global buffer_count
    global sent_data
    global sent_count
    global start_recieve
    global model
    global dictionary
    if buffer_count < buffer_length:
        buffer_data[buffer_count] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
        buffer_count += 1
        print 'not enough'
    else:
        mean = np.mean(buffer_data[:,0])
        now_data = abs(float(raw_data['FFA2']))
        if ((abs(now_data-mean) > bound) | start_recieve == 1):
            buffer_data[:buffer_length-1] = buffer_data[1:buffer_length]
            buffer_data[buffer_length-1] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
            sent_data[sent_count] = [float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])]
            print len(sent_data[sent_count]),len(buffer_data[buffer_length-1])
            sent_count += 1
            start_recieve = 1
            if sent_count == 50:
                testing_data = pd.DataFrame(sent_data, columns=['Axis1', 'Axis2', 'Axis3', 'Axis4', 'Axis5', 'Axis6'])
                result = train.Predicting(model, testing_data, dictionary)
                sent_count = 0
                start_recieve = 0
                print testing_data
                print result
            #print mean, now_data
            #print raw_data['FFA2'], raw_data['FFA3'], raw_data['FFA4'], raw_data['FFA6'], raw_data['FFA7'], raw_data['FFA8']


class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        json_data = self.request[0]
        raw_data = json.loads(json_data)
        direct_to_model(raw_data)


def start_server(name):
    print 'current ip address: ' + HOST
    cut_size = 50
    buffer_data = np.zeros([5, 6])
    buffer_count = 0
    sent_data = np.zeros([cut_size,6])
    sent_count = 0
    start_recieve = 0
    
    global buffer_data
    global buffer_count
    global sent_data
    global sent_count
    global start_recieve

    data = train.Load(name)
    training_features, labels, dictionary = train.Train_Preprocessing(data, cut_size=cut_size, slide_size=50, sample_ratio=0.2)
    model = train.Training(np.array(training_features), labels)
    global dictionary
    global model
    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
    server.serve_forever()
    
    

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "<Usage: <User1> <User2> <User3>>"
        exit()
    name = []
    for i in range(1, len(sys.argv)):
        name.append(sys.argv[i])
    
    start_server(name)
