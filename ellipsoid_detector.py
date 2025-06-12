#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Цитата для использования данного кода:
Aysan Mahmoudzadeh, Iman Azimi, Amir M. Rahmani, and Pasi Liljeberg, 
"Lightweight Photoplethysmography Quality Assessment for Real-time IoT-based Health Monitoring 
using Unsupervised Anomaly Detection," Elsevier International Conference on Ambient Systems, 
Networks and Technologies (ANT'21), 2021, Poland.
"""

import os
import sys
import wfdb
import numpy as np
import zipfile
import tempfile
from scipy import signal, stats
import pandas as pd
from glob import glob
from sklearn.covariance import EllipticEnvelope

# ====================== ФИЛЬТРАЦИЯ СИГНАЛА ======================
def butter_filtering(sig, fs, fc, order, btype):
    """
    Баттервортовский фильтр с улучшенной обработкой ошибок
    """
    print("\n=== butter_filtering ===")
    print(f"Входные данные: len(sig)={len(sig)}, fs={fs}, fc={fc}, order={order}, btype={btype}")
    
    # Проверка на NaN во входном сигнале
    if np.isnan(sig).any():
        print("Ошибка: входной сигнал содержит NaN!")
        return sig
        
    try:
        w = np.array(fc)/(fs/2)
        print(f"Нормализованные частоты: {w}")
        
        # Уменьшаем порядок фильтра, если возникают проблемы
        effective_order = min(order, 6)  # Не более 6 порядка
        if effective_order != order:
            print(f"Уменьшен порядок фильтра с {order} до {effective_order} для стабильности")
            
        b, a = signal.butter(effective_order, w, btype=btype, analog=False)
        
        # Проверка устойчивости фильтра
        if not np.all(np.abs(np.roots(a)) < 1):
            print("Предупреждение: фильтр потенциально неустойчив!")
            
        filtered = signal.filtfilt(b, a, sig)
        
        # Проверка результата
        if np.isnan(filtered).any():
            print("Ошибка: результат фильтрации содержит NaN! Возвращаем исходный сигнал")
            return sig
            
        print(f"Успешная фильтрация. Среднее: {np.mean(filtered):.4f}, STD: {np.std(filtered):.4f}")
        return filtered
        
    except Exception as e:
        print(f"Ошибка фильтрации: {str(e)}. Возвращаем исходный сигнал")
        return sig

# ====================== ДЕТЕКЦИЯ ПИКОВ ======================
def peak_detection(sig, fs):
    """
    Обнаружение пиков сердечных сокращений в сигнале PPG
    Параметры:
        sig - сигнал PPG
        fs - частота дискретизации (Гц)
    Возвращает:
        индексы пиков в сигнале
    """
    print("\n=== peak_detection ===")
    print(f"Входные данные: len(sig)={len(sig)}, fs={fs}")
    print(f"Первые 5 значений сигнала: {sig[:5]}")
    
    # 1. Первичное обнаружение пиков в сыром сигнале
    NN_index_sig = np.array(signal.argrelextrema(sig, np.greater)).reshape(1,-1)[0]
    print(f"Найдено {len(NN_index_sig)} пиков в сыром сигнале")
    if len(NN_index_sig) > 0:
        print(f"Первые 5 индексов пиков: {NN_index_sig[:5]}")
        print(f"Значения в пиках: {sig[NN_index_sig[:5]]}")
    
    # 2. Анализ спектра для определения частоты сердцебиения
    f, ppg_den = signal.periodogram(sig, fs)
    min_f = np.where(f >= 0.6)[0][0]  # мин. частота пульса (0.6 Гц = 36 уд/мин)
    max_f = np.where(f >= 3.0)[0][0]  # макс. частота пульса (3.0 Гц = 180 уд/мин)
    ppg_HR_freq = ppg_den[min_f:max_f]
    HR_freq = f[min_f:max_f]
    
    print(f"Диапазон частот пульса: {HR_freq[0]:.2f}-{HR_freq[-1]:.2f} Гц")
    
    # 3. Определение доминирующей частоты пульса
    HRf = HR_freq[np.argmax(ppg_HR_freq)]
    print(f"Доминирующая частота пульса: {HRf:.2f} Гц ({HRf*60:.1f} уд/мин)")
    
    # 4. Настройка границ полосы фильтра
    boundary = 0.5  # ±0.5 Гц вокруг доминирующей частоты
    HRfmin = max(HRf - boundary, 0.6)
    HRfmax = min(HRf + boundary, 3.0)
    print(f"Границы полосы фильтра: {HRfmin:.2f}-{HRfmax:.2f} Гц")
    
    # 5. Фильтрация сигнала в диапазоне пульса
    filtered = butter_filtering(sig, fs, np.array([HRfmin, HRfmax]), 4, 'bandpass')
    
    # 6. Обнаружение пиков в отфильтрованном сигнале
    NN_index_filtered = np.array(signal.argrelextrema(filtered, np.greater)).reshape(1,-1)[0]
    print(f"Найдено {len(NN_index_filtered)} пиков в отфильтрованном сигнале")
    if len(NN_index_filtered) > 0:
        print(f"Первые 5 индексов пиков: {NN_index_filtered[:5]}")
        print(f"Значения в пиках: {filtered[NN_index_filtered[:5]]}")
    
    # 7. Сопоставление пиков в сыром и отфильтрованном сигналах
    NN_index = np.array([]).astype(int)
    for i in NN_index_filtered:
        NN_index = np.append(NN_index, NN_index_sig[np.abs(i - NN_index_sig).argmin()])
    
    NN_index = np.unique(NN_index)
    print(f"Итоговое количество уникальных пиков: {len(NN_index)}")
    if len(NN_index) > 0:
        print(f"Первые 5 итоговых пиков: {NN_index[:5]}")
        print(f"Значения в итоговых пиках: {sig[NN_index[:5]]}")
    
    return NN_index

# ====================== ДЕТЕКЦИЯ ВПАДИН ======================
def troughs_detection(sig, NN_index):
    """
    Обнаружение впадин (минимальных точек) между пиками
    Параметры:
        sig - сигнал PPG
        NN_index - индексы пиков
    Возвращает:
        индексы впадин
    """
    print("\n=== troughs_detection ===")
    print(f"Входные данные: len(sig)={len(sig)}, количество пиков={len(NN_index)}")
    if len(NN_index) > 0:
        print(f"Первые 5 пиков: {NN_index[:5]}")
    
    MM_index = np.array([]).astype(int)
    for i in range(NN_index.shape[0]-1):
        start = NN_index[i]
        end = NN_index[i+1]
        segment = sig[start:end]
        if len(segment) == 0:
            print(f"Предупреждение: пустой сегмент между пиками {i} и {i+1}")
            continue
        min_idx = np.argmin(segment) + start
        MM_index = np.append(MM_index, min_idx)
    
    print(f"Найдено {len(MM_index)} впадин")
    if len(MM_index) > 0:
        print(f"Первые 5 впадин: {MM_index[:5]}")
        print(f"Значения в впадинах: {sig[MM_index[:5]]}")
    
    return MM_index

# ====================== РАСЧЕТ ЭНТРОПИЙ ======================
def approx_entropy(U, m, r) -> float:
    """
    Расчет приближенной энтропии (ApEn) - меры регулярности сигнала
    Параметры:
        U - временной ряд
        m - длина шаблона (обычно 2)
        r - порог схожести (обычно 0.1-0.2 от STD)
    """
    print("\n=== approx_entropy ===")
    print(f"Входные данные: len(U)={len(U)}, m={m}, r={r:.4f}")
    if len(U) < m + 1:
        print("Предупреждение: длина сигнала слишком мала для расчета энтропии")
        return np.nan
    
    U = np.array(U)
    N = U.shape[0]
    
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        if z == 0:
            print("Ошибка: z=0 в _phi")
            return 10
        else:
            X = np.repeat(x[:, np.newaxis], 1, axis=2)
            C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
            return np.log(C).sum() / z
    
    result = abs(_phi(m + 1) - _phi(m))
    print(f"Результат approx_entropy: {result:.4f}")
    return result

def shan_entropy(sig):
    """
    Расчет энтропии Шеннона
    """
    print("\n=== shan_entropy ===")
    print(f"Входные данные: len(sig)={len(sig)}")
    if len(sig) == 0:
        print("Ошибка: пустой входной сигнал")
        return np.nan
    
    hist = np.histogram(sig, 10)[0]  # 10 бинов гистограммы
    print(f"Гистограмма: {hist}")
    pk = hist/hist.sum()
    result = stats.entropy(pk, base=2)
    print(f"Результат shan_entropy: {result:.4f}")
    return result

def spec_entropy(sig, fs):
    """
    Расчет спектральной энтропии
    """
    print("\n=== spec_entropy ===")
    print(f"Входные данные: len(sig)={len(sig)}, fs={fs}")
    if len(sig) == 0:
        print("Ошибка: пустой входной сигнал")
        return np.nan
    
    _, Pxx_den = signal.welch(sig, fs)  # Оценка спектральной плотности
    pk = Pxx_den/np.sum(Pxx_den)
    result = stats.entropy(pk, base=2)
    print(f"Результат spec_entropy: {result:.4f}")
    return result

# ====================== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ======================
def feature_extraction(sig, fs):
    """
    Извлечение 5 ключевых признаков из сигнала PPG с проверкой на пустые значения
    """
    print("\n=== feature_extraction ===")
    print(f"Входные данные: len(sig)={len(sig)}, fs={fs}")
    
    try:
        # 1. Обнаружение пиков и впадин
        peaks_idx = peak_detection(sig, fs)
        if len(peaks_idx) < 2:  # Нужно минимум 2 пика для анализа
            print("Ошибка: недостаточно пиков (<2) для анализа")
            return [np.nan] * 5
            
        troughs_idx = troughs_detection(sig, peaks_idx)
        if len(troughs_idx) < 2:  # Нужно минимум 2 впадины
            print("Ошибка: недостаточно впадин (<2) для анализа")
            return [np.nan] * 5
            
        # 2. Расчет признаков для каждого сердечного цикла
        f_skewness = []
        f_kurtosis = []
        f_approx_entropy = []
        
        for i in range(len(troughs_idx)-1):
            heart_cycle = sig[troughs_idx[i]:troughs_idx[i+1]]
            if len(heart_cycle) < 10:  # Проверка на пустой цикл
                print(f"Предупреждение: пустой сердечный цикл между впадинами {i} и {i+1}")
                continue
                
            print(f"\nОбработка сердечного цикла {i+1}/{len(troughs_idx)-1}")
            print(f"Длина цикла: {len(heart_cycle)} отсчетов")
            
            skew = stats.skew(heart_cycle)
            kurt = stats.kurtosis(heart_cycle)
            ape = approx_entropy(heart_cycle, 2, 0.2*np.std(heart_cycle))
            
            print(f"Асимметрия: {skew:.4f}")
            print(f"Эксцесс: {kurt:.4f}")
            print(f"Приближенная энтропия: {ape:.4f}")
            
            f_skewness.append(skew)
            f_kurtosis.append(kurt)
            f_approx_entropy.append(ape)
        
        # 3. Проверка и расчет вариаций признаков
        if not f_skewness:
            print("Ошибка: не удалось рассчитать асимметрию")
            f_skewness = np.nan
        else:
            f_skewness = np.nanmax(f_skewness) - np.nanmin(f_skewness)
            
        if not f_kurtosis:
            print("Ошибка: не удалось рассчитать эксцесс")
            f_kurtosis = np.nan
        else:
            f_kurtosis = np.nanmax(f_kurtosis) - np.nanmin(f_kurtosis)
            
        if not f_approx_entropy:
            print("Ошибка: не удалось рассчитать приближенную энтропию")
            f_approx_entropy = np.nan
        else:
            f_approx_entropy = np.nanmax(f_approx_entropy) - np.nanmin(f_approx_entropy)
        
        # 4. Расчет энтропийных признаков
        f_shan_entropy = shan_entropy(sig)
        f_spec_entropy = spec_entropy(sig, fs)
        
        result = [f_skewness, f_kurtosis, f_approx_entropy, f_shan_entropy, f_spec_entropy]
        print("\nИтоговые признаки:")
        print(f"1. Асимметрия: {result[0]:.4f}")
        print(f"2. Эксцесс: {result[1]:.4f}")
        print(f"3. Приближенная энтропия: {result[2]:.4f}")
        print(f"4. Энтропия Шеннона: {result[3]:.4f}")
        print(f"5. Спектральная энтропия: {result[4]:.4f}")
        
        return result
    
    except Exception as e:
        print(f"Ошибка в feature_extraction: {str(e)}")
        return [np.nan] * 5

def parse_hea_file(hea_content):
    """
    Парсит содержимое .hea файла и извлекает параметры преобразования для PPG сигнала
    Возвращает (scale, offset)
    """
    for line in hea_content.split('\n'):
        if 'wrist_ppg' in line:
            # Пример строки: s1_walk.dat 16 29.6893(-43403)/mV 0 0 5035 30732 0 wrist_ppg
            parts = line.split()
            scale_offset_part = parts[2]  # Часть вида "29.6893(-43403)/mV"
            
            # Извлекаем scale и offset
            scale_str, rest = scale_offset_part.split('(', 1)
            offset_str = rest.split(')', 1)[0]
            
            scale = float(scale_str)
            offset = float(offset_str)
            
            return scale, offset
    
    raise ValueError("Не удалось найти параметры преобразования для PPG в .hea файле")

# ====================== ОСНОВНАЯ ПРОГРАММА ======================
if __name__ == '__main__':
    # Конфигурация
    fs = 256.0  # Частота дискретизации (Гц)
    zip_path = "wrist-ppg-during-exercise-1.0.0.zip"  # Путь к архиву
    zip_internal_dir = "wrist-ppg-during-exercise-1.0.0/"  # Папка внутри архива

    # 1. Поиск всех уникальных записей (без расширений) в ZIP
    record_files = set()
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for f in zip_ref.namelist():
            if f.startswith(zip_internal_dir) and (f.endswith('.dat') or f.endswith('.hea')):
                base_name = os.path.splitext(os.path.basename(f))[0]
                record_files.add(base_name)
    
    if not record_files:
        sys.exit('No WFDB records found in the ZIP archive')
    
    print(f"\nНайдены следующие записи: {sorted(record_files)}")

    # 2. Обработка файлов
    feature_list = []
    
    for record_name in sorted(record_files):
        try:
            print(f"\n\n====== Обработка записи {record_name} ======")
            
            # Создаем временную папку для WFDB
            with tempfile.TemporaryDirectory() as temp_dir:
                # Извлекаем нужные файлы (.dat и .hea) во временную папку
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for ext in ['.dat', '.hea']:
                        zip_file_path = f"{zip_internal_dir}{record_name}{ext}"
                        if zip_file_path in zip_ref.namelist():
                            # Извлекаем с сохранением структуры
                            zip_ref.extract(zip_file_path, temp_dir)
                            # Перемещаем файлы в корень временной папки
                            extracted_path = os.path.join(temp_dir, zip_file_path)
                            new_path = os.path.join(temp_dir, f"{record_name}{ext}")
                            os.rename(extracted_path, new_path)
                
                # Читаем .hea файл для получения параметров преобразования
                hea_path = os.path.join(temp_dir, f"{record_name}.hea")
                with open(hea_path, 'r') as f:
                    hea_content = f.read()
                
                scale, offset = parse_hea_file(hea_content)
                print(f"Параметры преобразования: scale={scale}, offset={offset}")
                
                # Загружаем запись из временной папки
                record_path = os.path.join(temp_dir, record_name)
                record = wfdb.rdrecord(record_path)
                
                # Извлечение данных PPG (первый канал)
                ppg_raw = record.p_signal[:, 0]
                ppg_mv = (ppg_raw - offset) / scale
                ppg_filtered = butter_filtering(ppg_mv, fs, [0.6, 3.0], 4, 'bandpass')
                
                # Проверочный вывод
                print(f"\n--- Проверка записи {record_name} ---")
                print(f"Количество отсчетов: {len(ppg_raw)}")
                print(f"Частота дискретизации: {record.fs} Hz")
                print("Первые 10 значений сырого сигнала:")
                print(ppg_raw[:10])
                print("Первые 10 значений после масштабирования:")
                print(ppg_mv[:10])
                print("Первые 10 значений после фильтрации:")
                print(ppg_filtered[:10])

                # Извлечение признаков
                features = feature_extraction(ppg_filtered, fs)
                feature_list.append(features)
                
                print(f"\n✓ Успешно обработан: {record_name}")
                
        except Exception as e:
            print(f"\n× Ошибка при обработке {record_name}: {str(e)}")
            continue
    
    
    # 3. Обучение модели Elliptic Envelope
    # Нормализация данных
    X_train_mean = np.mean(np.array(feature_list), axis=0)
    X_train_std = np.std(np.array(feature_list), axis=0)
    X_train = (np.array(feature_list) - X_train_mean)/X_train_std
    
    # Создание и обучение модели
    model_EE = EllipticEnvelope(contamination=0.34)  # Ожидается 34% выбросов
    model_EE.fit(X_train)
    
    # 4. Тестирование модели (пример)
    # X_test = (np.array(feature_list) - X_train_mean)/X_train_std
    # yhat = model_EE.predict(X_test)
    # Результат: 1 для надежных сигналов, -1 для ненадежных