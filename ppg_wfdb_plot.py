import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def read_wfdb_record(record_path):
    """
    Чтение записи WFDB из файлов .hea, .dat, .atr
    """
    # Проверка существования файлов
    required_files = [f"{record_path}.hea", f"{record_path}.dat"]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Не найден файл: {file}")
    
    # Чтение записи
    record = wfdb.rdrecord(record_path)
    annotations = wfdb.rdann(record_path, 'atr') if os.path.exists(f"{record_path}.atr") else None
    
    return record, annotations

def butter_bandpass_filter(data, fs, lowcut=0.6, highcut=3.0, order=1):
    """
    Бандасс-фильтр Баттерворта для PPG сигнала
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def partial_filter(data, fs, alpha=0.3):
    """
    Частичная фильтрация (alpha = степень фильтрации 0-1)
    """
    filtered = butter_bandpass_filter(data, fs, order=1)
    return alpha*filtered + (1-alpha)*data

def plot_ppg_from_wfdb(record_path, show_plot=True, save_plot=False, output_file='ppg_plot.png'):
    """
    Построение графика PPG сигнала из файлов WFDB
    
    Параметры:
    record_path (str): Путь к записи (без расширения)
    show_plot (bool): Показывать график
    save_plot (bool): Сохранять график в файл
    output_file (str): Имя файла для сохранения
    """
    try:
        # Чтение данных
        record, annotations = read_wfdb_record(record_path)
        
        # Получение параметров сигнала
        fs = record.fs  # Частота дискретизации
        ppg_raw = record.p_signal[:, 0]  # Первый канал (PPG)
        signal_length = len(ppg_raw)
        time = np.arange(signal_length) / fs
        
        # Фильтрация сигнала
        ppg_filtered = partial_filter(ppg_raw, fs)
        
        # Создание графика
        plt.figure(figsize=(14, 8))
        
        # Сырой сигнал
        plt.subplot(2, 1, 1)
        plt.plot(time, ppg_raw, color='blue', linewidth=0.5)
        plt.title('Raw PPG Signal', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(10, 20)  # Ограничение для сырого сигнала
        
        # Фильтрованный сигнал
        plt.subplot(2, 1, 2)
        plt.plot(time, ppg_filtered, color='red', linewidth=0.8)
        plt.title('Filtered PPG Signal (0.6-3.0 Hz)', fontsize=12)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(10, 20)  # Ограничение для фильтрованного сигнала
        
        plt.tight_layout() # Магическое исправление всех перекрытий
        
        # Сохранение графика
        if save_plot:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"График сохранен как {output_file}")
        
        # Показать график
        if show_plot:
            plt.show()
            
    except Exception as e:
        print(f"Ошибка: {str(e)}")

# Пример использования
if __name__ == "__main__":
    # Укажите путь к вашей записи (без расширения)
    record_path = "data_wfdb/s1_walk"  # Например "100" если файлы называются 100.hea, 100.dat
    
    # Построение графиков
    plot_ppg_from_wfdb(record_path, 
                      show_plot=True,
                      save_plot=True,
                      output_file="ppg_wfdb_plot.png")
