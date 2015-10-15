from pubnub import Pubnub

__author__ = 'maeglin89273'

PUB_KEY = 'pub-c-7b8779c0-5d71-42e8-9041-b91b538ec2b4'
SUB_KEY = 'sub-c-57d15590-2530-11e5-b6a9-0619f8945a4f'
CHANNEL = 'door_channel'

def direct_to_model(raw_data):
     print raw_data

def callback(message, channel):
   direct_to_model(message)


def error(message):
    print 'ERROR : ' + str(message)


def connect(message):
    print 'CONNECTED'

def reconnect(message):
    print 'RECONNECTED'


def disconnect(message):
    print 'DISCONNECTED'

if __name__ == '__main__':
    pubnub = Pubnub(publish_key=PUB_KEY, subscribe_key=SUB_KEY)
    pubnub.subscribe(channels=CHANNEL, callback=callback, error=error,
                     connect=connect, reconnect=reconnect, disconnect=disconnect)
