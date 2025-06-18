import os
import pickle

import forecasting
import numpy as np
import pandas as pd
from replace import analyze_with_heartpy, plot_signals, replace_segments
from shared import load_ppg


def load_trained_model(model_path="nf_model.pkl"):
    """
    Пытается загрузить ранее сохранённый объект NeuralForecast из файла.
    Если файл не найден или не удалось загрузить, возвращает None.
    """
    if not os.path.exists(model_path):
        print(f"[Model] Файл модели '{model_path}' не найден.")
        return None
    try:
        with open(model_path, "rb") as f:
            nf = pickle.load(f)
        print(f"[Model] Модель загружена из {model_path}")
        return nf
    except Exception as e:
        print(f"[Model] Ошибка загрузки модели из '{model_path}': {e}")
        return None


def train_and_save_model_if_needed(train_filenames, data_path, model_path, repeats=10):
    """
    Если в forecasting есть функция train_and_save_model, вызывает её на train_filenames.
    После сохранения загружает модель.
    Возвращает загруженный объект или None.
    """
    if hasattr(forecasting, "train_and_save_model"):
        try:
            print("[Model] Обучение модели через forecasting.train_and_save_model...")
            forecasting.train_and_save_model(
                train_filenames, data_path, model_path, repeats=repeats
            )
            nf = load_trained_model(model_path)
            if nf is not None:
                print("[Model] Модель обучена и загружена")
                return nf
        except Exception as e:
            print(f"[Model] Ошибка при обучении модели: {e}")
    else:
        print("[Model] forecasting.train_and_save_model не реализован.")
    return None


def fallback_train_and_get_model(train_filenames, data_path, repeats=10):
    """
    Фоллбэк: если нет train_and_save_model, но есть forecast_signal,
    вызывает forecast_signal на train_filenames, обучая модель на лету,
    и возвращает объект nf, полученный внутри.
    Предполагаем, что forecasting.forecast_signal возвращает dict name->forecast_df,
    но нам нужен сам объект nf. Если forecast_signal не возвращает nf,
    можно модифицировать forecasting.py, чтобы после выполнения fit он сохранял nf глобально.
    Здесь мы демонстрируем простой подход: вызываем forecast_signal и затем пытаемся загрузить nf
    через forecasting.load_last_model() или похожую функцию, если она появится.
    Если же ничего нет, возвращаем None.
    """
    if hasattr(forecasting, "forecast_signal"):
        try:
            print(
                "[Fallback] Запуск forecasting.forecast_signal на тренировочных данных..."
            )
            _ = forecasting.forecast_signal(
                train_filenames, data_path=data_path, repeats=repeats
            )
            # Попытка загрузить модель после этого:
            if hasattr(forecasting, "load_last_model"):
                nf = forecasting.load_last_model()
                if nf is not None:
                    print(
                        "[Fallback] Модель получена через forecasting.load_last_model"
                    )
                    return nf
            # Если нет функции load_last_model, и forecasting не возвращает nf,
            # то на данном этапе мы не имеем прямого доступа к nf.
            print(
                "[Fallback] Нет способа получить nf из forecast_signal. Переходим без модели."
            )
        except Exception as e:
            print(f"[Fallback] Ошибка при fallback-прогнозе: {e}")
    else:
        print("[Fallback] forecasting.forecast_signal не реализован.")
    return None


def find_bad_segments_simple(original, predicted, threshold_factor=2.0):
    """
    Простая детекция аномалий: сравнение целиком оригинала и предсказания.
    Возвращает список сегментов [(start, end), ...]
    """
    L = min(len(original), len(predicted))
    orig = original[:L]
    pred = predicted[:L]
    error = np.abs(orig - pred)
    std = np.std(error)
    if std == 0:
        std = 1e-6
    bad_mask = error > threshold_factor * std
    segments = []
    in_seg = False
    for i, flag in enumerate(bad_mask):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            end = i
            segments.append((start, end))
            in_seg = False
    if in_seg:
        segments.append((start, L))
    return segments


def sliding_window_anomaly_detection(
    signal, nf, name, fs, N, K, step, threshold_factor
):
    """
    Скользящая оконная детекция аномалий используя обученную модель nf.
    Возвращает (bad_segments, avg_errors).
    Предполагается, что nf.predict(df_window) выдаёт DataFrame с колонкой 'NBEATS'.
    """
    M = len(signal)
    errors = np.zeros(M)
    counts = np.zeros(M, dtype=int)
    # Базовое время для ds
    base_time = pd.Timestamp("2025-01-01")
    # Шаг времени в ms
    if fs is not None and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20
    for t in range(N, M - K + 1, step):
        window = signal[t - N : t]
        # Формируем ds для окна: начиная от base_time + (t-N)*freq_ms
        start_time = base_time + pd.to_timedelta((t - N) * freq_ms, unit="ms")
        ds = pd.date_range(start=start_time, periods=N, freq=f"{freq_ms}ms")
        df_win = pd.DataFrame({"unique_id": name, "ds": ds, "y": window})
        try:
            if hasattr(forecasting, "predict_window"):
                forecast_df = forecasting.predict_window(nf, df_win)
            else:
                forecast_df = nf.predict(df_win)
        except Exception as e:
            print(f"[Detect] Ошибка прогноза при t={t}: {e}")
            continue
        if "NBEATS" not in forecast_df.columns:
            continue
        pred_vals = forecast_df["NBEATS"].values
        if len(pred_vals) < K:
            continue
        for i in range(K):
            idx = t + i
            if idx >= M:
                break
            err = abs(pred_vals[i] - signal[idx])
            errors[idx] += err
            counts[idx] += 1
    avg_errors = np.zeros(M)
    mask = counts > 0
    avg_errors[mask] = errors[mask] / counts[mask]
    overall_std = np.std(avg_errors[mask]) if np.any(mask) else 1e-6
    bad_mask = avg_errors > threshold_factor * overall_std
    segments = []
    in_seg = False
    for i, flag in enumerate(bad_mask):
        if flag and not in_seg:
            start = i
            in_seg = True
        elif not flag and in_seg:
            end = i
            segments.append((start, end))
            in_seg = False
    if in_seg:
        segments.append((start, M))
    return segments, avg_errors


def generate_good_segments(signal, nf, name, bad_segments, fs, N):
    """
    Для каждого bad_segment (b0,b1) формирует прогноз из истории длины N перед b0.
    Возвращает список good_segments.
    """
    good_segments = []
    base_time = pd.Timestamp("2025-01-01")
    if fs is not None and fs > 0:
        freq_ms = int(round(1000.0 / fs))
    else:
        freq_ms = 20
    for b0, b1 in bad_segments:
        if b0 < N:
            continue
        window = signal[b0 - N : b0]
        # Формируем ds: для простоты можно начать с base_time, т.к. относительная разница одинакова
        ds = pd.date_range(start=base_time, periods=N, freq=f"{freq_ms}ms")
        df_win = pd.DataFrame({"unique_id": name, "ds": ds, "y": window})
        try:
            if hasattr(forecasting, "predict_window"):
                forecast_df = forecasting.predict_window(nf, df_win)
            else:
                forecast_df = nf.predict(df_win)
        except Exception as e:
            print(f"[Replace] Ошибка прогноза для замены сегмента {b0}-{b1}: {e}")
            continue
        if "NBEATS" not in forecast_df.columns:
            continue
        pred_vals = forecast_df["NBEATS"].values
        L = b1 - b0
        if len(pred_vals) < L:
            continue
        good_segments.append(pred_vals[:L])
    return good_segments


def process_single_record(
    name, nf, forecast_results, data_path, use_fallback, N, K, step, threshold_factor
):
    """
    Обрабатывает одну запись: загрузка, детекция аномалий, генерация и замена, визуализация и анализ.
    """
    print(f"\n=== Обработка записи {name} ===")
    # 1. Загрузка и (возможно) предобработка
    try:
        signal, fs = load_ppg(f"{data_path}/{name}")
        if hasattr(forecasting, "load_and_preprocess"):
            signal = forecasting.load_and_preprocess(f"{data_path}/{name}")
    except Exception as e:
        print(f"[Main] Ошибка загрузки сигнала {name}: {e}")
        return

    # 2. Детекция
    if not use_fallback and nf is not None:
        bad_segments, avg_errors = sliding_window_anomaly_detection(
            signal, nf, name, fs, N, K, step, threshold_factor
        )
    else:
        # fallback: простое сравнение с глобальным прогнозом (если есть)
        if name in forecast_results:
            forecast_df = forecast_results[name]
            if "NBEATS" in forecast_df.columns and not forecast_df.empty:
                pred = forecast_df["NBEATS"].values
                bad_segments = find_bad_segments_simple(signal, pred, threshold_factor)
                avg_errors = None
            else:
                bad_segments, avg_errors = [], None
        else:
            bad_segments, avg_errors = [], None

    print(f"[Main] Найдено {len(bad_segments)} плохих сегментов: {bad_segments}")

    # 3. Генерация хороших сегментов
    good_segments = []
    if bad_segments:
        if not use_fallback and nf is not None:
            good_segments = generate_good_segments(
                signal, nf, name, bad_segments, fs, N
            )
        else:
            # В fallback: можно брать из глобального прогноза, если прогноз длиннее
            for b0, b1 in bad_segments:
                forecast_df = forecast_results.get(name)
                if forecast_df is None or "NBEATS" not in forecast_df.columns:
                    continue
                pred = forecast_df["NBEATS"].values
                L = b1 - b0
                if len(pred) >= b1:
                    good_segments.append(pred[b0:b1])

    # 4. Замена
    if good_segments and bad_segments:
        replaced = replace_segments(signal, bad_segments, good_segments)
    else:
        replaced = signal.copy()
        print("[Main] Нет сегментов для замены или не удалось сгенерировать прогноз")

    # 5. Визуализация
    try:
        plot_signals(signal, replaced, fs, bad_segments)
    except Exception as e:
        print(f"[Main] Ошибка визуализации {name}: {e}")

    # 6. Анализ HeartPy
    try:
        analyze_with_heartpy(signal, fs, replaced)
    except Exception as e:
        print(f"[Main] Ошибка HeartPy-анализа {name}: {e}")


def main():
    # 1) Параметры
    train_filenames = ["s1_walk", "s2_walk", "s3_walk"]
    test_filename = "s6_walk"
    data_path = "../test_data"
    model_path = "nf_model.pkl"
    repeats = 10

    # 2) Попытка загрузить модель
    nf = load_trained_model(model_path)
    if nf is None:
        # Пытаемся обучить и сохранить
        nf = train_and_save_model_if_needed(
            train_filenames, data_path, model_path, repeats=repeats
        )
    # Если всё ещё нет модели, fallback
    use_fallback = nf is None

    # 3) Если fallback, получить глобальный прогноз для train_filenames (или для test? Нам нужен прогноз по test)
    forecast_results = {}
    if use_fallback:
        # Обучаем "на лету" только на train_filenames, затем прогнозируем test_filename
        print(
            "[Fallback] Обучаем на тренировочных данных и прогнозируем тестовую запись"
        )
        # forecast_signal обычно прогнозирует для всех указанных имён. Передадим train+test?
        # Но лучше: так как модель обучается на train_filenames внутри, и потом прогнозируем test.
        # Если forecast_signal не поддерживает отдельный режим train/test, можно:
        #  - вызвать forecast_signal(train_filenames) чтобы обучить, но не получить объект
        #  - затем внутри forecasting не будет кэша nf, поэтому предсказание test не работает напрямую.
        # Поэтому в fallback ограничимся прогнозом test как в старом варианте:
        #   forecast_signal([test_filename]) — но тогда обучится только на test, что не совсем то.
        # Лучше: просим forecast_signal сразу на train_filenames+test_filename,
        #   но тогда он обучит на всех, включая тест, что не идеально, но ближе к старому.
        all_names = train_filenames + [test_filename]
        try:
            forecast_results = forecasting.forecast_signal(
                all_names, data_path=data_path, repeats=repeats
            )
        except Exception as e:
            print(f"[Fallback] Ошибка при попытке прогнозирования train+test: {e}")
            forecast_results = {}
    else:
        # nf загружен, ничего не делаем здесь — при обработке test_filename будем вызывать nf.predict
        forecast_results = {}

    # 4) Параметры детекции
    N = getattr(forecasting, "INPUT_SIZE", 512)
    K = getattr(forecasting, "HORIZON", 100)
    step = getattr(forecasting, "DETECTION_STEP", 50)
    threshold_factor = getattr(forecasting, "THRESHOLD_FACTOR", 2.0)

    # 5) Обработка только тестовой записи
    process_single_record(
        test_filename,
        nf,
        forecast_results,
        data_path,
        use_fallback,
        N,
        K,
        step,
        threshold_factor,
    )

    print("\n=== Обработка завершена ===")


if __name__ == "__main__":
    main()
