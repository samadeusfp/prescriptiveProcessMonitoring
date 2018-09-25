import csv
import os
from sys import argv



dataset_name = argv[1]
preds_dir = argv[2]
fileName = os.path.join(preds_dir, "preds_val_%s.csv" % dataset_name)

if dataset_name.startswith("traffic"):
    positiveMultiplier = 1.02
    negativeMultiplier = 0.98
else:
    positiveMultiplier = 1.002
    negativeMultiplier = 0.998

events = []
with open(fileName, newline='') as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter=';')
    for row in csvReader:
        events.append(row)

with open(fileName, 'w', newline='') as csvResultFile:
    csvWriter = csv.writer(csvResultFile, delimiter=';', quoting=csv.QUOTE_NONE)
    firstLine = True
    currentCaseId = ""
    currentPreds = 0
    for row in events:
        if row[1] == currentCaseId:
            if row[0] == "1":
                row[2] = float(currentPreds) * positiveMultiplier
            else:
                row[2] = float(currentPreds) * negativeMultiplier
        print(repr(row))
        currentCaseId = row[1]
        currentPreds = row[2]
        csvWriter.writerow(row)
