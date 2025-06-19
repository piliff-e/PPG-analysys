import wfdb
import numpy as np
import pandas as pd
from scipy.signal import decimate
from neuralforecast import NeuralForecast
from scipy.signal import savgol_filter
from pyts.metrics import dtw
from forecasting import recursive_forecast

record = wfdb.rdrecord('../test_data/s10_sit')
signal = decimate(record.p_signal[:, 0], q=5)
signal = savgol_filter(signal, 15, 3)
signal = (signal - np.mean(signal)) / np.std(signal)

signal_to_compare = decimate(record.p_signal[513:1013, 0], q=5)
signal_to_compare = savgol_filter(signal_to_compare, 15, 3)
signal_to_compare = (signal_to_compare - np.mean(signal_to_compare)) / np.std(signal_to_compare)

compare_df = pd.DataFrame({
    'unique_id': "signal_to_compare",
    'ds': pd.date_range(start='2025-01-01', periods=len(signal_to_compare), freq='20ms'),
    'y': signal_to_compare
})

df = pd.DataFrame({
    'unique_id': "signal",
    'ds': pd.date_range(start='2025-01-01', periods=len(signal), freq='20ms'),
    'y': signal
})

nf = NeuralForecast.load('../models/v1')  
forecast_df = recursive_forecast(nf, df, repeats=5)


forecast_to_compare = forecast_df['NBEATS']
alignment = dtw(forecast_to_compare, signal_to_compare)
print(f'DTW Distance: {alignment}')












