#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.covariance import EllipticEnvelope

def read_wfdb_record(record_path):
    """Чтение записи WFDB из файлов .hea, .dat, .atr"""
    required_files = [f"{record_path}.hea", f"{record_path}.dat"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Не найден файл: {file}")
    
    record = wfdb.rdrecord(record_path)
    return record

def butter_filtering(sig, fs, fc, order, btype): 
    """Фильтрация сигнала фильтром Баттерворта"""
    w = np.array(fc)/(fs/2)
    b, a = signal.butter(order, w, btype=btype, analog=False)
    filtered = signal.filtfilt(b, a, sig)
    return filtered

def segment_ppg_signal(ppg_signal, fs, segment_length=10):
    """Разделение сигнала PPG на сегменты заданной длины (в секундах)"""
    samples_per_segment = int(fs * segment_length)
    num_segments = len(ppg_signal) // samples_per_segment
    segments = []
    
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = ppg_signal[start:end]
        segments.append(segment)
    
    return segments

def plot_segments(segments, fs, num_segments=10):
    """Визуализация первых N сегментов сигнала"""
    plt.figure(figsize=(12, 2*num_segments))
    
    for i in range(min(num_segments, len(segments))):
        segment = segments[i]
        time = np.arange(len(segment)) / fs
        
        plt.subplot(num_segments, 1, i+1)
        plt.plot(time, segment, color='blue', linewidth=0.8)
        plt.title(f'Сегмент {i+1} (0-10 секунд)', fontsize=10)
        plt.ylabel('Амплитуда')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if i != num_segments-1:
            plt.xticks([])
        else:
            plt.xlabel('Время (с)')
    
    plt.tight_layout()
    plt.show()

# ... [остальные функции остаются без изменений: peak_detection, troughs_detection, 
# approx_entropy, shan_entropy, spec_entropy, feature_extraction] ...

def peak_detection(sig, fs):
    """Обнаружение пиков в сигнале"""
    NN_index_sig = np.array(signal.argrelextrema(sig, np.greater)).reshape(1,-1)[0]
    
    f, ppg_den = signal.periodogram(sig, fs)
    min_f = np.where(f >= 0.6)[0][0]
    max_f = np.where(f >= 3.0)[0][0]
    ppg_HR_freq = ppg_den[min_f:max_f]
    HR_freq = f[min_f:max_f]
    
    HRf = HR_freq[np.argmax(ppg_HR_freq)]
    boundary = 0.5
    HRfmin = max(HRf - boundary, 0.6)
    HRfmax = min(HRf + boundary, 3.0)
    
    filtered = butter_filtering(sig, fs, np.array([HRfmin, HRfmax]), 5, 'bandpass')
    NN_index_filtered = np.array(signal.argrelextrema(filtered, np.greater)).reshape(1,-1)[0]
    
    NN_index = np.array([]).astype(int)
    for i in NN_index_filtered:
        NN_index = np.append(NN_index, NN_index_sig[np.abs(i - NN_index_sig).argmin()])
    NN_index = np.unique(NN_index)
    
    return NN_index

def troughs_detection(sig, NN_index):
    """Обнаружение впадин в сигнале"""
    MM_index = np.array([]).astype(int)
    for i in range(NN_index.shape[0]-1):
        MM_index = np.append(MM_index, np.argmin(sig[NN_index[i]:NN_index[i+1]]) + NN_index[i])
    return MM_index

def approx_entropy(U, m, r) -> float:
    """Вычисление аппроксимационной энтропии"""
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        if z == 0:
            return 10
        else:
            X = np.repeat(x[:, np.newaxis], 1, axis=2)
            C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
            return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))

def shan_entropy(sig):
    """Вычисление энтропии Шеннона"""
    hist = np.histogram(sig, 10)[0]
    pk = hist/hist.sum()
    return stats.entropy(pk, base=2)

def spec_entropy(sig, fs):   
    """Вычисление спектральной энтропии"""
    _, Pxx_den = signal.welch(sig, fs)
    pk = Pxx_den/np.sum(Pxx_den)
    return stats.entropy(pk, base=2)
    
def feature_extraction(sig, fs):
    """Извлечение признаков из сигнала"""
    peaks_idx = peak_detection(sig, fs)
    troughs_idx = troughs_detection(sig, peaks_idx)
    
    f_skewness = []
    f_kurtosis = []
    f_approx_entropy = []
    
    for i in range(troughs_idx.size-1):
        heart_cycle = sig[troughs_idx[i]:troughs_idx[i+1]]
        f_skewness.append(stats.skew(heart_cycle))
        f_kurtosis.append(stats.kurtosis(heart_cycle))
        f_approx_entropy.append(approx_entropy(heart_cycle, 2, 0.1*np.std(heart_cycle)))
    
    f_skewness = max(f_skewness)-min(f_skewness)
    f_kurtosis = max(f_kurtosis)-min(f_kurtosis)
    f_approx_entropy = max(f_approx_entropy)-min(f_approx_entropy)
    f_shan_entropy = shan_entropy(sig)
    f_spec_entropy = spec_entropy(sig, fs)
    
    return [f_skewness, f_kurtosis, f_approx_entropy, f_shan_entropy, f_spec_entropy]




if __name__ == '__main__':
    fs = 20.0  # Частота дискретизации (Hz)
    load_dir = "data_wfdb/"  # Папка с WFDB файлами
    
    # Получаем список уникальных записей (без расширений)
    files = os.listdir(load_dir)
    record_names = set([f.split('.')[0] for f in files if f.endswith(('.dat', '.hea', '.atr'))])
    
    if not record_names:
        sys.exit('No WFDB files found in the directory')
    
    feature_list = []
    all_segments = []  # Для хранения всех сегментов
    
    for record_name in record_names:
        try:
            # Чтение записи
            record = read_wfdb_record(os.path.join(load_dir, record_name))
            ppg_signal = record.p_signal[:, 0]  # Предполагаем, что PPG в первом канале
            
            # Фильтрация сигнала
            ppg_filtered = butter_filtering(ppg_signal, fs, [0.6, 3.0], 5, 'bandpass')
            
            # Разделение на segment_length-секундные сегменты
            segments = segment_ppg_signal(ppg_filtered, fs, segment_length=10)
            all_segments.extend(segments)
            
            # Извлечение признаков из каждого сегмента
            for segment in segments:
                features = feature_extraction(segment, fs)
                feature_list.append(features)
                
        except Exception as e:
            print(f"Error processing {record_name}: {str(e)}")
            continue
    
    # Визуализация первых 10 сегментов
    if len(all_segments) >= 10:
        print("\nВизуализация первых 10 сегментов PPG сигнала (по 10 секунд каждый):")
        plot_segments(all_segments, fs, num_segments=10)
    else:
        print(f"\nДоступно только {len(all_segments)} сегментов для визуализации")
        plot_segments(all_segments, fs, num_segments=len(all_segments))
    
    # Обучение модели
    if not feature_list:
        sys.exit('No features extracted - check your data')
    
    X_train_mean = np.mean(np.array(feature_list), axis=0)
    X_train_std = np.std(np.array(feature_list), axis=0)
    X_train = (np.array(feature_list) - X_train_mean) / X_train_std
    
    model_EE = EllipticEnvelope(contamination=0.34)
    model_EE.fit(X_train)
    
    # Тестирование модели
    X_test = (np.array(feature_list) - X_train_mean) / X_train_std
    yhat = model_EE.predict(X_test)
    print("\nРезультаты классификации модели:")
    print("1 = нормальный сегмент, -1 = аномальный сегмент")
    print(yhat)
