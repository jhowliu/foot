import serial
from time import sleep
port = "/dev/cu.usbserial-AJ038LU9"
ser = serial.Serial(port, 9600, timeout=0)
while True:
    data = ser.read(9999)
    if len(data) > 0:
        print 'Got:', data
    sleep(0.5)
    print 'not blocked'

ser.close()
