import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import wfdb


# ---------- Загрузка ----------
def load_ppg(record_name):
    record = wfdb.rdrecord(record_name)
    signal = record.p_signal[:, 0]
    fs = record.fs
    return signal, fs


# ---------- Детекция аномалий через forecasting ----------
def detect_bad_intervals_by_forecast(signal, model, N, K, threshold, step=1):
    """
    Возвращает список bad_intervals по алгоритму скользящего окна прогнозирования.
    - signal: numpy array, полный сигнал.
    - model: объект с методом predict(window) -> array length K.
    - N, K: параметры модели.
    - threshold: порог усреднённой ошибки.
    - step: сдвиг окна (в отсчётах). Можно >1 для ускорения.
    """
    M = len(signal)
    errors = np.zeros(M, dtype=float)
    counts = np.zeros(M, dtype=int)

    # Проходим по окнам
    for t in range(N, M - K + 1, step):
        window = signal[t - N : t]
        # Нормализация окна, фильтрация и т.д. должны делаться внутри model.predict или до вызова
        pred = model.predict(window)  # ожидаем length K
        # Если predict возвращает list, конвертировать в np.array
        pred = np.asarray(pred)
        # Сравниваем
        for i in range(K):
            idx = t + i
            if idx >= M:
                break
            err = abs(pred[i] - signal[idx])
            errors[idx] += err
            counts[idx] += 1

    # Усредняем ошибки
    avg_errors = np.zeros(M, dtype=float)
    nonzero = counts > 0
    avg_errors[nonzero] = errors[nonzero] / counts[nonzero]
    # Теперь строим булев массив anomalous
    anomalous = avg_errors > threshold

    # Сгруппировать True в интервалы
    bad_intervals = []
    in_int = False
    for i, flag in enumerate(anomalous):
        if flag:
            if not in_int:
                start = i
                in_int = True
        else:
            if in_int:
                end = i
                bad_intervals.append((start, end))
                in_int = False
    if in_int:
        bad_intervals.append((start, M))
    return bad_intervals, avg_errors


# ---------- Генерация хорошего сегмента ----------
def generate_good_segment(signal, b0, b1, model, N, K):
    L = b1 - b0
    # Контекст до начала
    if b0 < N:
        context = signal[0:b0]
        context = np.pad(context, (N - len(context), 0), mode="edge")
    else:
        context = signal[b0 - N : b0]
    # Генерация
    if L <= K:
        pred = model.predict(context)
        good = np.asarray(pred)[:L]
    else:
        parts = []
        prev_context = context.copy()
        remaining = L
        # Предупреждение: качество предсказаний внутрь аномалии падает
        while remaining > 0:
            pred = model.predict(prev_context)
            pred = np.asarray(pred)
            take = min(remaining, K)
            parts.append(pred[:take])
            # Обновляем окно: берём последние N точек из prev_context + pred[:take]
            combined = np.concatenate([prev_context, pred[:take]])
            prev_context = combined[-N:]
            remaining -= take
        good = np.concatenate(parts)
    return good


# ---------- Замена сегментов ----------
def replace_segments(signal, bad_segments, good_segments):
    out = signal.copy()
    for (b0, b1), good in zip(bad_segments, good_segments):
        L = b1 - b0
        good = np.asarray(good)
        if len(good) != L:
            # интерполяция внутри bad-участка
            good = np.interp(
                np.linspace(0, len(good), L, endpoint=False), np.arange(len(good)), good
            )
        out[b0:b1] = good
    return out


# ---------- Пример работы ----------
def main():
    # Параметры модели forecasting
    N = 200  # количество точек на вход
    K = 50  # прогноз на K точек
    threshold = 0.1  # пример порога ошибки, нужно подобрать на валидации
    step = 5  # сдвиг окна для быстродействия

    # Загрузка сигнала
    signal, fs = load_ppg("path_to_record")  # заменить на реальный путь
    # Здесь должна быть фильтрация/нормализация, аналогичная той, на которой обучали модель
    # signal = preprocess(signal)

    # Загрузка/инициализация или передача обученной модели
    model = (
        ...
    )  # объект, у которого есть метод predict(window: array length N) -> array length K

    # 1) Обнаружение “плохих” интервалов
    bad_intervals, avg_errors = detect_bad_intervals_by_forecast(
        signal, model, N, K, threshold, step
    )

    print("Найденные плохие интервалы:", bad_intervals)

    # 2) Для каждого “плохого” интервала генерируем “хороший” и заменяем
    good_segments = []
    for b0, b1 in bad_intervals:
        good = generate_good_segment(signal, b0, b1, model, N, K)
        good_segments.append(good)
    replaced = replace_segments(signal, bad_intervals, good_segments)

    # 3) Визуализация
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="Оригинал")
    plt.plot(t, replaced, alpha=0.7, label="После замены")
    for b0, b1 in bad_intervals:
        plt.axvspan(b0 / fs, b1 / fs, color="red", alpha=0.2)
    plt.legend()
    plt.xlabel("Время (с)")
    plt.title("PPG: замена плохих сегментов по forecasting-анализу")
    plt.tight_layout()
    plt.show()

    # 4) HeartPy-аналитика
    try:
        wd_orig, m_orig = hp.process(signal, sample_rate=fs)
        wd_rep, m_rep = hp.process(replaced, sample_rate=fs)
        print("Original BPM:", m_orig.get("bpm"), "→ Replaced BPM:", m_rep.get("bpm"))
    except Exception as e:
        print("HeartPy анализ невозможен:", e)


if __name__ == "__main__":
    main()
