import numpy as np
import json

with open('data/hydrated/all.json') as fp:
	tweets = json.load(fp)

data = np.zeros((len(tweets),))
for i, tweet in enumerate(tweets):
	data[i] = tweet['label']

np.random.shuffle(data)

split = int(len(tweets)*0.8)
train = data[:split]
test = data[split:]

mn = train.mean()
mse = ((test-mn)**2).mean()

print(f'mean: {mn}; MSE: {mse}')
