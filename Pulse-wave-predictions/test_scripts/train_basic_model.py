import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import heartpy as hp
import wfdb
from scipy.signal import decimate
from neuralforecast.models import RNN, NBEATS
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter


all_dfs = []
filenames = ['s10_sit', 's11_sit', 's12_sit']
for filename in filenames:
    record = wfdb.rdrecord(f'../test_data/{filename}')
    signal = decimate(record.p_signal[:, 0], q=5)
    signal = savgol_filter(signal, 15, 3)
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    segment_df = pd.DataFrame({
        'unique_id': filename,
        'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
        'y': signal
    })
    all_dfs.append(segment_df)

df = pd.concat(all_dfs).reset_index(drop=True)

nf = NeuralForecast(
    models=[NBEATS(input_size=512, h=100, max_steps=500, learning_rate=1e-3,batch_size=8)],  # 128 прошлых точек → 50 шагов вперёд
    freq='20ms'
)
#ЧТОБЫ ДООБУЧИТЬ РАССКОММЕНТИРУЙ
# nf.models[0].load('../models/v2/version_2.pth')  # Загрузка модели из файла
for i in range(15):
    nf.fit(df)

# path_to_save = '../models/version_2.pth'
nf.save(path='../models/', model_index=[0], overwrite=True)
# nf.models[0].save(path_to_save)

