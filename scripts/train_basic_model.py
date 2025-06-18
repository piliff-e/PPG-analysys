import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from scipy.signal import decimate
from neuralforecast.models import NBEATS
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
    models=[NBEATS(input_size=500, h=100, max_steps=500, learning_rate=1e-3,batch_size=16)], 
    freq='20ms'
)

#ЧТОБЫ ДООБУЧИТЬ РАССКОММЕНТИРУЙ И УКАЖИ ПУТЬ К НУЖНОЙ МОДЕЛИ
# nf = NeuralForecast.load('../models/v1')  

for i in range(10):
    nf.fit(df)

nf.save(path='../models/v3', model_index=[0], overwrite=True)

