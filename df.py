import requests
import csv

res = requests.get('https://api.exmo.com/v1.1/candles_history?symbol=BTC_USD&resolution=30&from=1580556979&to=1585557979')
data = res.json()['candles']

with open('df.csv', 'w') as csvfile:
    fieldnames = ['t','v', 'c', 'o', 'l', 'h']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(data)

print("writing complete")