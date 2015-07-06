import sys
import pandas as pd

def Parse(filename):
    data = pd.read_csv(filename)
    data['Label'] = data['Label.1']

    data['time'] = map(lambda x, y, z: x * 3600 + y*60 + z, data['date_hour'], data['date_minute'], data['date_second'])

    data = data.drop(['date_hour', 'date_minute', 'date_second', 'Label.1'], 1)

    # Stable sorting algorithm
    data = data.sort('time', kind='mergesort')

    data.to_csv(filename, index=False)

# May be use time/sequence interval to cut
def CutByTime(data, interval):
    first = data['time'][0]

    result = []
    tmp = [[data['Axis1'][0], data['Axis2'][0], data['Axis3'][0]]]

    for (x, y, z, t) in zip(data['Axis1'][1:], data['Axis2'][1:], data['Axis3'][1:], data['time'][1:]):
        if (t - first) > interval:
            result.append(tmp)
            first = t
            tmp = []

        tmp.append([x, y, z])

    result.append(tmp)

    return result

# Yield n-sized chunks from data
def CutBySeq(data, n):
    result = []
    for i in xrange(0, len(data), n):
        result.append(data[i:i+n])

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'USAGE: python <python-file> <input-file>'
        exit()

    parse(sys.argv[1])
