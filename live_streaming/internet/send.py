import serial
port = "/dev/cu.usbserial-AJ038LU9"
ser = serial.Serial(port, 9600)
while True:
    x = ser.write('hello\n')

ser.close()
