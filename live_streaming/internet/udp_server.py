import SocketServer
import json
import sys
import numpy as np
import pandas as pd
__author__ = 'maeglin89273'

import socket as sk
sys.path.append('/Users/Terry/Work/foot/')
import train_dtw_demo as train
HOST = '192.168.0.2'
PORT = 3070

BUFFER_SIZE = 512


def direct_to_model(raw_data):
    bound = 0.4
    global buffer_length
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
        print 'build buffer data'
    else:
        mean = np.mean(buffer_data[:,0])
        now_data = abs(float(raw_data['FFA2']))
        if ((abs(now_data-mean) > bound) | start_recieve == 1):
            #print "start receive"
            buffer_data[:buffer_length-1] = buffer_data[1:buffer_length]
            buffer_data[buffer_length-1] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
            sent_data[sent_count] = [float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])]
            #print len(sent_data[sent_count]),len(buffer_data[buffer_length-1])
            sent_count += 1
            start_recieve = 1
            if sent_count == cut_size*5-1:
                print "start predict"
                testing_data = pd.DataFrame(sent_data, columns=['Axis1', 'Axis2', 'Axis3', 'Axis4', 'Axis5', 'Axis6'])
                result = train.Predicting(model, testing_data, dictionary, pca_model)
                buffer_data[:sent_count - cut_size] = buffer_data[cut_size:]
                sent_count -= cut_size
                start_recieve = 0
                #print testing_data
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
    cut_size = 100
    buffer_length = 10
    buffer_data = np.zeros([buffer_length, 6])
    buffer_count = 0
    sent_data = np.zeros([cut_size,6])
    sent_count = 0
    start_recieve = 0
    
    global cut_size
    global buffer_length
    global buffer_data
    global buffer_count
    global sent_data
    global sent_count
    global start_recieve
    data = train.Load(name)
    training_features, labels, dictionary,pca_model = train.Train_Preprocessing(data[:], cut_size=cut_size, slide_size=50, sample_ratio=0.8)
    print "Predicting"
    model = train.Training(np.array(training_features), labels)
    
    #print "fucking len"
    #print len(data[-1])
    #tmp = train.Cut(train.FuzzyDirection(data[-1])[0], 50)
    #tmp = train.FuzzyDirection(train.Cut(data[-1], 50)[:-1])
    
    #print len(data[-1]), len(data[-1])/50,len(tmp)
    global dictionary
    global model
    global pca_model
    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
    server.serve_forever()
    
    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "<Usage: <User1> <User2> <User3>>"
        exit()
    name = []
    for i in range(1, len(sys.argv)):
        name.append(sys.argv[i])
    
    start_server(name)
