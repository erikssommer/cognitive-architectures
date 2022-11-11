import csv
import sys

def readFiles(filename):
    with open(f'../../data/{filename}.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        # Skip the first line
        next(readCSV)

        inputs = []
        desired = []
        for row in readCSV:
            inputs.append([float(row[0]), float(row[1])])
            desired.append(float(row[2]))
        
        return inputs, desired

if __name__ == '__main__':
    filename = sys.argv[1]

    inputs, desired = readFiles(filename)

    