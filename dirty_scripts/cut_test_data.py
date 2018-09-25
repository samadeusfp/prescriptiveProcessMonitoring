import csv

firstLine = True
with open('/Users/stephanf/DataSets/traffic_fines_1.csv') as csvDataFile:
    with open('/Users/stephanf/DataSets/preprocessed_event_logs_prescriptive/traffic_fines_2.csv','w', newline='') as csvResultFile:
        csvReader = csv.reader(csvDataFile,delimiter=';', quoting=csv.QUOTE_NONE)
        csvWriter = csv.writer(csvResultFile, delimiter=';', quoting=csv.QUOTE_NONE)
        for row in csvReader:
            print(row)
            if firstLine == True:
                firstLine = False
                csvWriter.writerow(row)
            else:
                if row[4].startswith("A1") or row[4].startswith("V9"):
                    csvWriter.writerow(row)
print("Finished with cutting data")

