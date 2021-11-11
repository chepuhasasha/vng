import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Цвета граффиков
mpl.rcParams['figure.figsize'] = (15, 6)
mpl.rcParams['axes.facecolor'] = '#11121B'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = '#222433'

# Насктройки
TRAIN_SPLIT = 100000


# Импорт датасета и его обрезка.
df = pd.read_csv('./df.csv')[3000000:]

# Фильтрация NaN значений (обязательна)
df = df.dropna()

# Преобразование дат
df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df.head()