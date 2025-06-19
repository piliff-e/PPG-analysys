import numpy as np
import wfdb
from neuralforecast import NeuralForecast
from scipy.signal import decimate, savgol_filter


def load_and_preprocess(
    record_name: str,
    data_path: str,
    decimate_q: int = 5,
    sg_window: int = 15,
    sg_poly: int = 3,
) -> tuple[np.ndarray, float]:
    """
    Загружает PPG-сигнал через wfdb, децимирует, фильтрует Savitzky-Golay и нормализует.
    - record_name: имя записи без расширения, например 's13_sit'
    - data_path: путь к папке с WFDB-файлами, например 'test_data'
    - decimate_q: коэффициент децимации (если исходно fs=500, decimate_q=5 даст fs=100).
                  В идеале подобрать так, чтобы после децимации fs совпадал с тем, на котором обучена модель.
    - sg_window, sg_poly: параметры Savitzky-Golay. Применяется, если длина >= sg_window.
    Возвращает: (signal, fs) — numpy-массив нормализованного сигнала и новая частота fs.
    """
    # Загрузка через wfdb
    record = wfdb.rdrecord(f"{data_path}/{record_name}")
    raw = record.p_signal[:, 0]
    fs_raw = record.fs

    # Децимация
    if decimate_q and decimate_q > 1:
        signal = decimate(raw, q=decimate_q)
        fs = fs_raw / decimate_q
    else:
        signal = raw.copy()
        fs = fs_raw

    # Savitzky-Golay фильтрация (если применимо)
    if sg_window and sg_poly and len(signal) >= sg_window:
        try:
            signal = savgol_filter(signal, sg_window, sg_poly)
        except ValueError:
            # Например, если sg_window неправильно подобран
            pass

    # Нормализация: вычитаем среднее, делим на std (если std=0, делим на 1)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        std = 1.0
    signal = (signal - mean) / std

    return signal, fs


def load_model(model_path: str) -> NeuralForecast | None:
    """
    Загружает модель NeuralForecast из директории model_path.
    Возвращает загруженный объект или None при ошибке.
    """
    try:
        nf = NeuralForecast.load(model_path)
        print(f"[main] Модель загружена из '{model_path}'")
        return nf
    except Exception as e:
        print(f"[main] Ошибка загрузки модели из '{model_path}': {e}")
        return None
