import csv
import sys

def readFiles(filename):
    with open(f'../../data/{filename}.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        # Skip the first line
        next(readCSV)

        data = []
        for row in readCSV:
            

            data.append([float(row[0]), float(row[1]), float(row[2])])
        return data

if __name__ == '__main__':
    input = sys.argv[1]

    data = readFiles(input)

    print(data)