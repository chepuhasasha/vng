import requests
import csv
import datetime
import time
import sys

NAME = 'df'
SYMBOL = 'BTC_USD'  # пара
# конечное время (текушее время в секундах)
NOW = round(datetime.datetime.now().timestamp())
SIZE = 100000  # размер датасета
QUERY_SIZE = 1000  # сколько заберать за раз, API  имеет ограничение в 3000
TIMEOUT = 5  # задержка между запросами (секунды)
# 1, 5, 15, 30, 45, 60, 120, 180, 240 интервал между значениями (свеча) (минуты)
RESOLUTION = 5

DATA = []

RANGE = int(SIZE / QUERY_SIZE)
OFFSET = RESOLUTION * 60 * QUERY_SIZE
FROM = NOW - OFFSET
TO = NOW
for i in range(RANGE):
    res = requests.get('https://api.exmo.com/v1.1/candles_history', params={
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
    sys.stdout.write(
        '\r'+f'Выполнено запросов: {i+1} из {RANGE} || Получено строк: {(i+1)*QUERY_SIZE} из {SIZE}')
    sys.stdout.flush()
    # print(f'Получено строк: {(i+1)*QUERY_SIZE} из {SIZE}')
    # print(f'Выполнено запросов: {i+1} из {RANGE}')

# запись в csv
with open(f'{NAME}.csv', 'w') as csvfile:
    fieldnames = ['t', 'v', 'c', 'o', 'l', 'h']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(DATA)

sys.stdout.flush()
sys.stdout.write('\r'+'\n'+f'Имя: {NAME}.csv'+'\n'+f'Количество строк: {len(DATA) - 1}' +
                 '\n'+f'Время выполнения: {round(datetime.datetime.now().timestamp()- NOW)} секунд.')
sys.stdout.flush()
