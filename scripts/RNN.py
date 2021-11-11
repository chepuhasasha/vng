import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

CONFIG = {
    'DATA_FRAME_PATH': './df.csv',
    'DATA_FRAME_SPLIT': 3000000,
    'CONSIDER': ['Open', 'Close', 'High', 'Low'],
    'DATE_NAME': 'Timestamp',
    'TRAIN_SPLIT': 300000,
    'STEP': 60,
    'PAST_HISTORY': 720,
    'FUTURE_TARGET': 72

}

class RNN:
    def __init__(self, config):
        self.DATA_FRAME_PATH = config['DATA_FRAME_PATH']
        self.DATA_FRAME_SPLIT = config['DATA_FRAME_SPLIT']
        self.CONSIDER = config['CONSIDER']
        self.DATE_NAME = config['DATE_NAME']
        self.TRAIN_SPLIT = config['TRAIN_SPLIT']
        self.STEP = config['STEP']
        self.PAST_HISTORY = config['PAST_HISTORY']
        self.FUTURE_TARGET = config['FUTURE_TARGET']

    def START(self):
        self.SET_DF() # импорт и подготовка датафрейма
        self.STANDARDIZATION() # стандартизация данных
        self.PREPARATION() # подготовка обучающих данных (самая прожорливая)

    def SET_DF(self):
        # Импорт датасета и его обрезка.
        self.df = pd.read_csv(self.DATA_FRAME_PATH)[self.DATA_FRAME_SPLIT:]
        # Фильтрация NaN значений (обязательна)
        self.df = self.df.dropna()
        self.df.head()
        pass

    def STANDARDIZATION(self):
        # дата фрейм по интересуюшим нас влияющим параметрам
        self.features = self.df[self.CONSIDER]
        self.features.index = self.df[self.DATE_NAME]
        self.features.head()
        # стандартизация
        dataset = self.features.values
        data_mean = dataset[:self.TRAIN_SPLIT].mean(axis=0)
        data_std = dataset[:self.TRAIN_SPLIT].std(axis=0)
        self.dataset = (dataset - data_mean) / data_std

    def MULTI_DATA(self, dataset, target, start_index, end_index, history_size,
                   target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)

    def PREPARATION(self):
        # тренеровочные данные
        self.x_train_single, self.x_train_single = self.MULTI_DATA(dataset, dataset[:, 1], 0,
                                                          TRAIN_SPLIT, past_history,
                                                          future_target, STEP,
                                                          single_step=True)
        # валидируюшие данные
        self.x_val_single, self.y_val_single = self.MULTI_DATA(dataset, dataset[:, 1],
                                                      TRAIN_SPLIT, None, past_history,
                                                      future_target, STEP,
                                                      single_step=True)
        print ('Временной интервал: {}'.format(x_train_single[0].shape))
   def TRAIN(self):
        pass

    def SAVE_MODEL(self):
        pass

    def OPEN_MODEL(self):
        pass

    def PREDICTION(self):
        pass


n = RNN(CONFIG)
