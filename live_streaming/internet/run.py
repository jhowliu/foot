import SocketServer
import socket as sk
import json
import sys
import thread
import numpy as np
import pandas as pd
__author__ = 'maeglin89273'

import socket as sk
sys.path.append('../../')
import train_dtw_demo as train
HOST = "192.168.0.159"
PORT = 3070

BUFFER_SIZE = 1024


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
    try:
        parsed = [float(x) for x in raw_data['FFA2'].split(',')]
    except:
        return

    if buffer_count[slipper_no] < buffer_length:
        print 'collect buffer data'
        buffer_data_all[slipper_no][buffer_count[slipper_no]] = np.abs(parsed)
        buffer_count[slipper_no] += 1

    else:
        mean = np.mean(buffer_data_all[slipper_no][:,0])
        now_data = abs(parsed[0])
        #Over bound and start receive the data
        if ((abs(now_data-mean) > bound) | start_recieve[slipper_no] == 1):
            #Record the data for prediction model
            sent_data_all[slipper_no][sent_count[slipper_no]] = parsed
            #Record the data index
            sent_count[slipper_no] += 1
            start_recieve[slipper_no] = 1

            if first[slipper_no] == 1:
                print "over bound, ", slipper_no
                first[slipper_no] = 0

            if sent_count[slipper_no] == cut_size*cut_coef-1:
                print "start predict"
                testing_data = pd.DataFrame(sent_data_all[slipper_no], columns=['Axis1', 'Axis2', 'Axis3', 'Axis4', 'Axis5', 'Axis6'])
                result = train.Predicting(model[slipper_no], testing_data, dictionary, pca_model, cut_size, predict_slide_size)

                #Shift the sent_data about 1*cut_size to record the following data
                sent_data_all[slipper_no][:sent_count[slipper_no] - cut_size+1] = sent_data_all[slipper_no][cut_size:]
                sent_count[slipper_no] -= cut_size
                start_recieve[slipper_no] = 0

                first[slipper_no] = 1
                if result == 1:
                    result = slipper_no + 1
                print "The result of prediction: " + str(result)
                #message = str(slipper_no) + ',' + str(result) + '\n'
                #clientsocket.sendall(message)
        else:
            counter += 1
            if counter == 70:
                counter = 0
                #message = str(slipper_no) + '\n'
                #clientsocket.sendall(message)


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
        print "Disconnect the slippers " + self.slipper_addr_str


def start_server(name, member_num):
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
    global dictionary
    global model
    global pca_model

    print "MeM_Num:", member_num
    print 'current ip address: ' + HOST
    cut_coef = 10
    cut_size = 30
    slide_size = 30
    predict_slide_size = 20
    buffer_length = 10
    buffer_data_all = [np.zeros([buffer_length, 6]) for _ in xrange(member_num)]
    buffer_count = [0] * member_num
    sent_data_all = [np.zeros([cut_size*cut_coef, 6]) for _ in xrange(member_num)]

    sent_count = [0] * member_num
    start_recieve = [0] * member_num
    first = [1] * member_num
    counter = 0
    model = []

    #clientsocket = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    #clientsocket.connect((Server_Host, Server_Port))

    data = train.Load(name)
    training_features, labels, dictionary, pca_model = train.Train_Preprocessing(data[:], cut_size=cut_size, slide_size=slide_size, sample_ratio=0.7)
    #train.Ploting3D(training_features, labels)

    '''
    out_features = ['frank.csv', 'xing.csv', 'jhow.csv', 'terry.csv']
    for i in np.unique(labels):
        np.savetxt(out_features[i],np.array(training_features)[labels == i],delimiter=",")
    '''

    for i in range(member_num):
        sampling_features, sampling_labels = train.UnderSampling(training_features, labels, i)
        print "Model " + str(i)
        model.append(train.FindBestClf(np.array(sampling_features), sampling_labels, i))

    print "Ready to predict"
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "<Usage: <User1> <User2> <User3>>"
        exit()
    name = []
    for i in range(1, len(sys.argv)):
        name.append(sys.argv[i])

    start_server(name, member_num = (len(sys.argv)-1))
