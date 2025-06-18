# Модуль для поиска "плохих" сегментов PPG-сигнала на основе метода эллиптической оболочки (EllipticEnvelope)

import numpy as np
from scipy import signal, stats
from sklearn.covariance import EllipticEnvelope


def peak_detection(sig, fs):
    """
    Обнаружение пиков сердечных сокращений в PPG-сигнале.
    Возвращает индексы пиков.
    """
    # Первичное обнаружение экстремумов
    # Используем относительные максимумы
    try:
        peaks = np.array(signal.argrelextrema(sig, np.greater)).reshape(-1)
    except Exception:
        peaks = np.array([])
    # Анализ спектра для грубой оценки ЧСС
    if len(sig) == 0 or fs <= 0:
        return peaks
    f, pxx = signal.periodogram(sig, fs)
    # диапазон частот HR: от 0.6 до 3.0 Гц
    min_idx = np.searchsorted(f, 0.6)
    max_idx = np.searchsorted(f, 3.0)
    if max_idx <= min_idx or len(pxx) <= max_idx:
        return peaks
    hr_band = pxx[min_idx:max_idx]
    if len(hr_band) == 0:
        return peaks
    dom = np.argmax(hr_band)
    HRf = f[min_idx + dom]
    # Полоса ±0.5 Гц
    HRfmin = max(HRf - 0.5, 0.6)
    HRfmax = min(HRf + 0.5, 3.0)
    # Фильтрация полосовая
    # Нормализация частот
    try:
        w = np.array([HRfmin, HRfmax]) / (fs / 2)
        b, a = signal.butter(min(4, 6), w, btype="bandpass")
        filtered = signal.filtfilt(b, a, sig)
        # Повторное обнаружение экстремумов на отфильтрованном
        peaks_filt = np.array(signal.argrelextrema(filtered, np.greater)).reshape(-1)
        # Сопоставляем с исходными пиками: для каждого пика во filtered ищем ближайший в исходном
        if len(peaks) > 0 and len(peaks_filt) > 0:
            matched = []
            for i in peaks_filt:
                idx = peaks[np.abs(peaks - i).argmin()]
                matched.append(idx)
            peaks = np.unique(matched)
        elif len(peaks) == 0:
            peaks = peaks_filt
    except Exception:
        pass
    return peaks


def troughs_detection(sig, peaks_idx):
    """
    Обнаружение впадин между пиками.
    Возвращает индексы впадин.
    """
    troughs = []
    for i in range(len(peaks_idx) - 1):
        start = peaks_idx[i]
        end = peaks_idx[i + 1]
        if end <= start + 1:
            continue
        segment = sig[start:end]
        # минимальное значение
        rel = np.argmin(segment)
        troughs.append(start + rel)
    return np.array(troughs, dtype=int)


def approx_entropy(U, m, r):
    """Приближённая энтропия ApEn"""
    U = np.asarray(U)
    N = len(U)
    if N < m + 1:
        return np.nan

    def _phi(m):
        z = N - m + 1
        x = np.array([U[i : i + m] for i in range(z)])
        C = []
        for i in range(len(x)):
            # расстояние Chebyshev
            dist = np.max(np.abs(x - x[i]), axis=1)
            C.append(np.sum(dist <= r) / (z))
        C = np.array(C)
        # избегаем log(0)
        C = np.where(C > 0, C, 1e-10)
        return np.sum(np.log(C)) / (z)

    return abs(_phi(m + 1) - _phi(m))


def shan_entropy(sig, bins=10):
    """Энтропия Шеннона по гистограмме"""
    sig = np.asarray(sig)
    if len(sig) == 0:
        return np.nan
    hist, _ = np.histogram(sig, bins=bins)
    pk = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
    pk = np.where(pk > 0, pk, 1e-10)
    return stats.entropy(pk, base=2)


def spec_entropy(sig, fs):
    """Спектральная энтропия через Welch"""
    sig = np.asarray(sig)
    if len(sig) == 0 or fs <= 0:
        return np.nan
    f, Pxx = signal.welch(sig, fs)
    pk = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else np.zeros_like(Pxx)
    pk = np.where(pk > 0, pk, 1e-10)
    return stats.entropy(pk, base=2)


def extract_cycle_features(sig, fs):
    """
    Разбиение сигнала на сердечные циклы и извлечение признаков для каждого цикла.
    Возвращает:
      features: np.ndarray формы (num_cycles, num_features)
      intervals: список кортежей (start, end) индексов цикла в сигнале
    Признаки: [skewness, kurtosis, approx_entropy, shannon_entropy, spectral_entropy]
    """
    peaks = peak_detection(sig, fs)
    if len(peaks) < 2:
        return np.empty((0, 5)), []
    troughs = troughs_detection(sig, peaks)
    if len(troughs) < 2:
        return np.empty((0, 5)), []
    feats = []
    intervals = []
    for i in range(len(troughs) - 1):
        start = troughs[i]
        end = troughs[i + 1]
        cycle = sig[start:end]
        if len(cycle) < 10:
            continue
        skew = stats.skew(cycle)
        kurt = stats.kurtosis(cycle)
        ape = approx_entropy(
            cycle, 2, 0.2 * np.std(cycle) if np.std(cycle) > 0 else 0.1
        )
        sh = shan_entropy(cycle)
        se = spec_entropy(cycle, fs)
        feats.append([skew, kurt, ape, sh, se])
        intervals.append((start, end))
    if feats:
        return np.vstack(feats), intervals
    else:
        return np.empty((0, 5)), []


def train_envelope_model(feature_list, contamination=0.1):
    """
    Обучает EllipticEnvelope на списке признаков.
    feature_list: np.ndarray (num_samples, num_features).
    Возвращает (model, mean, std).
    """
    X = np.asarray(feature_list)
    # Удаляем строки с NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if len(X) == 0:
        raise ValueError("Нет корректных признаков для обучения")
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std > 0, std, 1.0)
    Xn = (X - mean) / std
    model = EllipticEnvelope(contamination=contamination)
    model.fit(Xn)
    return model, mean, std


def detect_bad_cycles(sig, fs, model, mean, std):
    """
    Детекция плохих циклов в сигнале с помощью обученного EllipticEnvelope.
    Возвращает список (start, end) плохих циклов.
    """
    features, intervals = extract_cycle_features(sig, fs)
    bad_intervals = []
    if features.size == 0:
        return bad_intervals
    # Нормализация
    Xn = (features - mean) / std
    # Обработка NaN после нормализации
    mask = ~np.isnan(Xn).any(axis=1)
    Xn_valid = Xn[mask]
    intervals_valid = [intervals[i] for i, ok in enumerate(mask) if ok]
    if len(Xn_valid) == 0:
        return bad_intervals
    labels = model.predict(Xn_valid)  # 1 или -1
    for lab, (s, e) in zip(labels, intervals_valid):
        if lab == -1:
            bad_intervals.append((s, e))
    # Объединяем смежные интервалы
    merged = []
    for interval in bad_intervals:
        if not merged:
            merged.append(list(interval))
        else:
            prev = merged[-1]
            if interval[0] <= prev[1]:
                prev[1] = max(prev[1], interval[1])
            else:
                merged.append(list(interval))
    return [tuple(x) for x in merged]


# Пример использования:
#
# from bad_segment_detector import extract_cycle_features, train_envelope_model, detect_bad_cycles
# signal, fs = load_ppg("../test_data/s1_walk")  # или предварительно предобработать
# # Собрать обучающие признаки из нескольких чистых сигналов:
# feature_list = []
# for name in ["s1_walk","s2_walk","s3_walk"]:
#     sig, fs = load_ppg(f"../test_data/{name}")
#     feats, _ = extract_cycle_features(sig, fs)
#     if feats.size>0:
#         feature_list.append(feats)
# if feature_list:
#     X_train = np.vstack(feature_list)
#     model, mean, std = train_envelope_model(X_train, contamination=0.1)
#     # Для тестовой записи:
#     sig_test, fs_test = load_ppg("../test_data/s6_walk")
#     bad = detect_bad_cycles(sig_test, fs_test, model, mean, std)
#     print("Найдены плохие интервалы:", bad)
#     # Далее замена циклов с помощью прогноза
