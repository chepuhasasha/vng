import requests
import csv
import datetime
import time
import os

NAME = 'df'
SYMBOL = 'BTC_USD' # пара
NOW = round(datetime.datetime.now().timestamp()) # конечное время (текушее время в секундах)
SIZE = 1000 # размер датасета
QUERY_SIZE = 100 # сколько заберать за раз, API  имеет ограничение в 3000
TIMEOUT = 1 # задержка между запросами (секунды)
RESOLUTION = 30 # 1, 5, 15, 30, 45, 60, 120, 180, 240 интервал между значениями (свеча) (минуты)

DATA = []

RANGE = int(SIZE / QUERY_SIZE)
OFFSET = RESOLUTION * 60 * QUERY_SIZE
FROM = NOW - OFFSET
TO = NOW
for i in range(RANGE):
  res = requests.get('https://api.exmo.com/v1.1/candles_history', params = {
    'symbol': SYMBOL,
    'resolution': RESOLUTION,
    'from': FROM,
    'to': TO
  })
  result = res.json()['candles']

  # обрезка лишних нулей
  for item in result:
    item['t'] = int(str(item['t'])[:-3])

  DATA = result + DATA

  # новый временной интервал
  FROM = DATA[0]['t'] - OFFSET
  TO = DATA[0]['t']

  time.sleep(TIMEOUT)
  os.system('clear')
  print(f'Получено строк: {(i+1)*QUERY_SIZE} из {SIZE}')
  print(f'Выполнено запросов: {i+1} из {RANGE}')

# запись в csv
with open(f'{NAME}.csv', 'w') as csvfile:
    fieldnames = ['t','v', 'c', 'o', 'l', 'h']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(DATA)

os.system('clear')
print('Датасет собран.')
print(f'Имя: {NAME}.csv')
print(f'Количество строк: {len(DATA) - 1}')
print(f'Время выполнения: {round(datetime.datetime.now().timestamp()- NOW)} секунд.')
