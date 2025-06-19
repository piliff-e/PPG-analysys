from forecasting import detect_and_replace_segments
from plotting import plot_mismatches
from preprocessing import load_and_preprocess, load_model


def main():
    # === 1. Настройки ===
    record_name = "s13_sit"
    data_path = "../test_data"
    model_path = "../models/v3"

    # Параметры предобработки
    decimate_q = 5
    sg_window = 15
    sg_poly = 3

    # Параметры прогнозирования и детекции
    window_size = 500  # N и одновременно шаг
    repeats = 5  # число итераций recursive_forecast
    threshold = 16.0  # DTW-порог (можно вычислить скриптом compute_distance.py)
    freq = "20ms"  # строка частоты, должна совпадать с обучением

    # === 2. Загрузка и предобработка сигнала ===
    signal, fs = load_and_preprocess(
        record_name,
        data_path,
        decimate_q=decimate_q,
        sg_window=sg_window,
        sg_poly=sg_poly,
    )
    print(f"[main] Загружен сигнал '{record_name}', длина={len(signal)}, fs={fs} Hz")
    orig_signal = signal.copy()  # резервная копия для отрисовки

    # === 3. Загрузка модели ===
    nf = load_model(model_path)
    if nf is None:
        return

    # === 4. Детекция и замена сегментов ===
    replaced_signal, bad_segments = detect_and_replace_segments(
        signal=signal,
        nf=nf,
        record_name=record_name,
        window_size=window_size,
        repeats=repeats,
        threshold=threshold,
        freq=freq,
    )
    if not bad_segments:
        print("[main] Плохие сегменты не найдены. Завершаем.")
    else:
        print(f"[main] Найдено и заменено сегментов: {bad_segments}")

    # === 5. Визуализация несовпадений ===
    plot_mismatches(orig_signal, replaced_signal, fs, bad_segments)


if __name__ == "__main__":
    main()
