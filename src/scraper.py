import requests

url = 'https://twitter.com/jlam55555/status/1299509919813644288'
page = requests.get(url)

print(page)