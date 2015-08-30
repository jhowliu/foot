import serial
import SocketServer
import socket as sk
import json
import sys
import thread
import numpy as np
import pandas as pd
__author__ = 'maeglin89273'

import socket as sk
sys.path.append('/Users/Terry/Work/foot/')
import train_dtw_demo as train
HOST = '192.168.0.2'
PORT = 3070

BUFFER_SIZE = 512
port = "/dev/cu.usbserial-AJ038LU9"
ser = serial.Serial(port, 9600)

def direct_to_model(raw_data):
    bound = 0.4
    global clientsocket
    global cut_coef
    global slide_size
    global cut_size
    global buffer_length
    global buffer_data_all
    global buffer_count
    global sent_data_all
    global sent_count
    global start_recieve
    global model
    global dictionary
    global first
    global counter

    slipper_no = int(raw_data['Label'])

    if buffer_count[slipper_no] < buffer_length:
        print 'collect buffer data'
        buffer_data_all[int(slipper_no)][buffer_count[slipper_no]] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
        buffer_count[slipper_no] += 1
    
    else:
        mean = np.mean(buffer_data_all[slipper_no][:,0])
        now_data = abs(float(raw_data['FFA2']))
        #Over bound and start receive the data
        if ((abs(now_data-mean) > bound) | start_recieve[slipper_no] == 1):
            
            #Save the data to buffer
            #buffer_data[:buffer_length-1] = buffer_data[1:buffer_length]
            #buffer_data[buffer_length-1] = np.abs([float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])])
            
            #Record the data for prediction model
            sent_data_all[slipper_no][sent_count] = [float(raw_data['FFA2']), float(raw_data['FFA3']), float(raw_data['FFA4']), float(raw_data['FFA6']), float(raw_data['FFA7']), float(raw_data['FFA8'])]
            #print len(sent_data[sent_count]),len(buffer_data[buffer_length-1])
            
            #Record the data index
            sent_count[slipper_no] += 1
            start_recieve[slipper_no] = 1
            
            if first[slipper_no] == 1:
                print "over bound, ", slipper_no
                first[slipper_no] = 0

            if sent_count[slipper_no] == cut_size*cut_coef-1:
                print "start predict"
                testing_data = pd.DataFrame(sent_data_all[slipper_no], columns=['Axis1', 'Axis2', 'Axis3', 'Axis4', 'Axis5', 'Axis6'])
                result = train.Predicting(model, testing_data, dictionary, pca_model, cut_size, predict_slide_size)
                
                #Shift the sent_data about 1*cut_size to record the following data
                sent_data_all[slipper_no][:sent_count[slipper_no] - cut_size+1] = sent_data_all[slipper_no][cut_size:]
                sent_count[slipper_no] -= cut_size

                start_recieve[slipper_no] = 0

                first[slipper_no] = 1
                    
                #print result
                
                message = '$' + str(slipper_no) + ',' + str(result) + '\n'
                ser.write(message)
                print message
                #clientsocket.sendall(message)
        #else:
            #print counter, "stop"
            #counter += 1
            #if counter == 50:
                #counter = 0
                #print "send_stop"
                #message = str(slipper_no) + '\n'
                #ser.write(message)
                #clientsocket.sendall(message)
    #Reach the num of records 
            


class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        json_data = self.request[0]
        raw_data = json.loads(json_data)
        direct_to_model(raw_data)

#def start_multithread(thread_num, name):
#    print 'Start: ', thread_num
#    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
#    server.serve_forever()

def start_server(name, member_num):
    print 'current ip address: ' + HOST
    Server_Host = '140.118.155.161'
    Server_Port = 15712
    cut_coef = 4
    cut_size = 50
    slide_size = 30
    predict_slide_size = 20
    buffer_length = 10
    buffer_data_all = [np.zeros([buffer_length, 6])] * member_num
    buffer_count = [0] * member_num
    sent_data_all = [np.zeros([cut_size*cut_coef,6])] * member_num
    sent_count = [0] * member_num
    start_recieve = [0] * member_num
    first = [1] * member_num
    counter = 0
    
    
    global clientsocket
    global counter   
    global first 
    global cut_coef
    global cut_size
    global predict_slide_size
    global buffer_length
    global buffer_data_all
    global buffer_count
    global sent_data_all
    global sent_count
    global start_recieve
    
    clientsocket = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    #clientsocket.connect((Server_Host, Server_Port))
    print "Connect to server: " + Server_Host
    
    data = train.Load(name)
    training_features, labels, dictionary,pca_model = train.Train_Preprocessing(data[:], cut_size=cut_size, slide_size=slide_size, sample_ratio=0.5)
    
    train.Ploting3D(training_features, labels)
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
    
    start_server(name, member_num = len(sys.argv))
