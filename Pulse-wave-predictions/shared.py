import wfdb


def load_ppg(record_name):
    """
    Загрузка PPG-сигнала из WFDB-файлов (.hea/.dat/.atr).
    """
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, 0]  # одномерный список float'ов
    fs = record.fs  # частота семплирования
    return signal, fs
