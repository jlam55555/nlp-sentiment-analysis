import json
import csv

labels = {}

with open('data/labels/all.csv') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		# some days don't have labels
		if len(row) < 2:
			continue

		labels[row[0]] = float(row[1])

outdata = []

with open('data/hydrated/all_unlabeled.json') as jsonfile:
	jsondata = json.load(jsonfile)

	for item in jsondata:
		if item['id'] in labels:
			item['label'] = labels[item['id']]
			outdata.append(item)
		
with open('data/hydrated/all.json', 'w+') as jsonfile:
	json.dump(outdata, jsonfile, indent=4)