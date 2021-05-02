import json
import matplotlib.pyplot as plt

with open('data/hydrated/all.json') as jsonfile:
	data = json.load(jsonfile)

bins = [0, 0, 0]
for i, l in enumerate(data):
	data[i] = l = l['label']
	if l < 0:
		bins[0] += 1
	elif l > 0:
		bins[2] += 1
	else:
		bins[1] += 1

print(f'Negative: {bins[0]}')
print(f'Zero:     {bins[1]}')
print(f'Positive: {bins[2]}')

plt.hist(data, bins=250)
plt.savefig('hist.png')